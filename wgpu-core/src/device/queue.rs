/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    command::{
        texture_copy_view_to_hal, validate_linear_texture_data, validate_texture_copy_range,
        CommandAllocator, CommandBuffer, CopySide, ImageCopyTexture, TransferError, BITS_PER_BYTE,
    },
    conv,
    device::{alloc, DeviceError, WaitIdleError},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id,
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::{Buffer, BufferAccessError, BufferMapState, BufferUse, Texture, TextureUse},
    FastHashMap, FastHashSet,
};
use core::{iter, mem, ops::Range, ptr, sync::atomic::Ordering};

use hal::{command::CommandBuffer as _, device::Device as _, queue::Queue as _};
use smallvec::SmallVec;
use thiserror::Error;

struct StagingData<B: hal::Backend> {
    buffer: B::Buffer,
    memory: alloc::MemoryBlock<B>,
}

#[derive(Debug)]
pub enum TempResource<B: hal::Backend> {
    Buffer(B::Buffer),
    Image(B::Image),
}

#[derive(Debug)]
pub(crate) struct PendingWrites<B: hal::Backend> {
    pub command_buffer: Option<B::CommandBuffer>,
    pub temp_resources: Vec<(TempResource<B>, alloc::MemoryBlock<B>)>,
    pub temp_buffer_resources: Vec<id::Valid<id::BufferId>>,
    pub dst_buffers: FastHashSet<id::BufferId>,
    pub dst_textures: FastHashSet<id::TextureId>,
}

impl<B: hal::Backend> PendingWrites<B> {
    pub fn new() -> Self {
        Self {
            command_buffer: None,
            temp_resources: Vec::new(),
            temp_buffer_resources: Vec::new(),
            dst_buffers: FastHashSet::default(),
            dst_textures: FastHashSet::default(),
        }
    }

    pub fn dispose(
        self,
        device: &B::Device,
        cmd_allocator: &CommandAllocator<B>,
        mem_allocator: &mut alloc::MemoryAllocator<B>,
    ) {
        if let Some(raw) = self.command_buffer {
            cmd_allocator.discard_internal(raw);
        }
        for (resource, memory) in self.temp_resources {
            mem_allocator.free(device, memory);
            match resource {
                TempResource::Buffer(buffer) => unsafe {
                    device.destroy_buffer(buffer);
                },
                TempResource::Image(image) => unsafe {
                    device.destroy_image(image);
                },
            }
        }
        // No need to explicitly handle temp_buffer_resources, as they will be disposed when
        // buffers are destroyed.
    }

    pub fn consume_temp(
        &mut self,
        resource: id::Valid<id::BufferId>, /*TempResource<B>, memory: alloc::MemoryBlock<B>*/
    ) {
        self.temp_buffer_resources
            .push(/*(resource, memory)*/ resource);
    }

    fn consume(&mut self, stage: StagingData<B>, cmdbuf: B::CommandBuffer) {
        self.temp_resources
            .push((TempResource::Buffer(stage.buffer), stage.memory));
        self.command_buffer = Some(cmdbuf);
    }

    #[must_use]
    fn finish(&mut self) -> Option<B::CommandBuffer> {
        self.dst_buffers.clear();
        self.dst_textures.clear();
        self.command_buffer.take().map(|mut cmd_buf| unsafe {
            cmd_buf.finish();
            cmd_buf
        })
    }

    pub(super) fn borrow_cmd_buf(
        &mut self,
        cmd_allocator: &CommandAllocator<B>,
    ) -> &mut B::CommandBuffer {
        if self.command_buffer.is_none() {
            let mut cmdbuf = cmd_allocator.allocate_internal();
            unsafe {
                cmdbuf.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
            }
            self.command_buffer = Some(cmdbuf);
        }
        self.command_buffer.as_mut().unwrap()
    }
}

#[derive(Default)]
struct RequiredBufferInits {
    map: FastHashMap<id::BufferId, Vec<Range<wgt::BufferAddress>>>,
}

impl RequiredBufferInits {
    fn add<B: hal::Backend>(
        &mut self,
        buffer_memory_init_actions: &[MemoryInitTrackerAction<id::BufferId>],
        buffer_guard: &mut Storage<Buffer<B>, id::BufferId>,
    ) -> Result<(), QueueSubmitError> {
        for buffer_use in buffer_memory_init_actions.iter() {
            let buffer = buffer_guard
                .get_mut(buffer_use.id)
                .map_err(|_| QueueSubmitError::DestroyedBuffer(buffer_use.id))?;

            let uninitialized_ranges = buffer.initialization_status.drain(buffer_use.range.clone());
            match buffer_use.kind {
                MemoryInitKind::ImplicitlyInitialized => {
                    uninitialized_ranges.for_each(drop);
                }
                MemoryInitKind::NeedsInitializedMemory => {
                    self.map
                        .entry(buffer_use.id)
                        .or_default()
                        .extend(uninitialized_ranges);
                }
            }
        }
        Ok(())
    }
}

impl<B: hal::Backend> super::Device<B> {
    fn prepare_stage(&self, size: wgt::BufferAddress) -> Result<StagingData<B>, DeviceError> {
        profiling::scope!("prepare_stage");
        let mut buffer = unsafe {
            self.raw
                .create_buffer(
                    size,
                    hal::buffer::Usage::TRANSFER_SRC,
                    hal::memory::SparseFlags::empty(),
                )
                .map_err(|err| match err {
                    hal::buffer::CreationError::OutOfMemory(_) => DeviceError::OutOfMemory,
                    _ => panic!("failed to create staging buffer: {}", err),
                })?
        };
        //TODO: do we need to transition into HOST_WRITE access first?
        let requirements = unsafe {
            self.raw.set_buffer_name(&mut buffer, "<write_buffer_temp>");
            self.raw.get_buffer_requirements(&buffer)
        };

        let block = self.mem_allocator.lock().allocate(
            &self.raw,
            requirements,
            gpu_alloc::UsageFlags::UPLOAD | gpu_alloc::UsageFlags::TRANSIENT,
        )?;
        block.bind_buffer(&self.raw, &mut buffer)?;
        Ok(StagingData {
            buffer,
            memory: block,
        })
    }

    fn post_prepare_stage_raw(&self, pending_writes: &mut PendingWrites<B>) -> B::CommandBuffer {
        match pending_writes.command_buffer.take() {
            Some(cmdbuf) => cmdbuf,
            None => {
                let mut cmdbuf = self.cmd_allocator.allocate_internal();
                unsafe {
                    cmdbuf.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
                }
                cmdbuf
            }
        }
    }

    fn initialize_buffer_memory(
        pending_writes: &mut PendingWrites<B>,
        mut required_buffer_inits: RequiredBufferInits,
        trackers: &mut crate::device::Trackers,
        buffer_guard: &mut Storage<Buffer<B>, id::BufferId>,
        cmd_allocator: &CommandAllocator<B>,
    ) -> Result<(), QueueSubmitError> {
        /*self.*/
        pending_writes
            .dst_buffers
            .extend(required_buffer_inits.map.keys());

        let cmd_buf = /*self.*/pending_writes.borrow_cmd_buf(cmd_allocator);
        /* let mut trackers = self.trackers.lock(); */

        for (buffer_id, mut ranges) in required_buffer_inits.map.drain() {
            // Collapse touching ranges. We can't do this any earlier since we only now gathered ranges from several different command buffers!
            ranges.sort_by(|a, b| a.start.cmp(&b.start));
            for i in (1..ranges.len()).rev() {
                assert!(ranges[i - 1].end <= ranges[i].start); // The memory init tracker made sure of this!
                if ranges[i].start == ranges[i - 1].end {
                    ranges[i - 1].end = ranges[i].end;
                    ranges.swap_remove(i); // Ordering not important at this point
                }
            }

            // Don't do use_replace since the buffer may already no longer have a ref_count.
            // However, we *know* that it is currently in use, so the tracker must already know about it.
            let transition = trackers.buffers.change_replace_tracked(
                id::Valid(buffer_id),
                (),
                BufferUse::COPY_DST,
            );
            let buffer = buffer_guard.get(buffer_id).unwrap();
            let &(ref buffer_raw, _) = buffer
                .raw
                .as_deref()
                .ok_or(QueueSubmitError::DestroyedBuffer(buffer_id))?;
            unsafe {
                cmd_buf.pipeline_barrier(
                    super::all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    transition.map(|pending| pending.into_hal(buffer_raw)),
                );
            }
            for range in ranges {
                let size = range.end - range.start;

                assert!(range.start % 4 == 0, "Buffer {:?} has an uninitialized range with a start not aligned to 4 (start was {})", buffer, range.start);
                assert!(size % 4 == 0, "Buffer {:?} has an uninitialized range with a size not aligned to 4 (size was {})", buffer, size);

                unsafe {
                    cmd_buf.fill_buffer(
                        buffer_raw,
                        hal::buffer::SubRange {
                            offset: range.start,
                            size: Some(size),
                        },
                        0,
                    );
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[error("queue is invalid")]
pub struct InvalidQueue;

#[derive(Clone, Debug, Error)]
pub enum QueueWriteError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
}

#[derive(Clone, Debug, Error)]
pub enum QueueSubmitError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error("buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("texture {0:?} is destroyed")]
    DestroyedTexture(id::TextureId),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("swap chain output was dropped before the command buffer got submitted")]
    SwapChainOutputDropped,
    #[error("GPU got stuck :(")]
    StuckGpu,
}

//TODO: move out common parts of write_xxx.

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    /// Safety: must synchronize with any calls to buffer_destroy, buffer_drop, or unmap for the
    /// same buffer.
    ///
    /// NOTE: Currently NOT marked unsafe to make it easier to integrate with the gfx_select!
    /// macro, but it should be marked unsafe if this is backported to non-Veloren wgpu.
    pub fn queue_write_buffer<'a, B: GfxBackend>(
        &'a self,
        queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_buffer", "Queue");

        let hub = B::hub(self);
        let mut token_ = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token_);
        let device = device_guard
            .get(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data.len() as wgt::BufferAddress,
                queued: true,
            });
        }

        let data_size = data.len() as wgt::BufferAddress;
        if data_size == 0 {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        }

        let mut stage = device.prepare_stage(data_size)?;
        stage.memory.write_bytes(&device.raw, 0, data)?;

        let (buffer_guard, mut token1) = hub.buffers.read(&mut token);
        let (mut trackers, token2) = token1.lock(&device.trackers);
        let (dst, transition) = trackers
            .buffers
            .use_replace(&*buffer_guard, buffer_id, (), BufferUse::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        drop((trackers, token2));
        let &(ref dst_raw, _) = dst
            .raw
            .as_deref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        let &Buffer {
            usage,
            size,
            ref map_state,
            ref life_guard,
            ..
        } = dst;
        // NOTE: Since we synchronize with unmap on this buffer, and also with the part of
        // queue_submit that calls [use_at] on buffers (which takes buffers mutably), we
        // know nobody else is concurrently modifying the submission index when we read it
        // here.
        let old_submission_index = life_guard.submission_index.load(Ordering::Relaxed);
        if matches!(map_state, BufferMapState::Init { .. }) && old_submission_index == 0 {
            // We have an unmapped map-on-init buffer.  Currently this means we always reject
            // writes until it's unmapped.  Since we synchronize with unmap, this is not racy.
            return Err(TransferError::InvalidBuffer(buffer_id).into());
        }
        // Safety: For the buffer to be deallocated, it must be unregistered.  All methods that
        // unregister the buffer, either require &mut on the device, or first require dropping
        // or destroying the buffer, which can't happen until we return due to the precondition
        // on this function.
        let dst_raw: &'a B::Buffer = unsafe { mem::transmute(dst_raw) };
        drop((buffer_guard, token1));

        if !usage.contains(wgt::BufferUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(Some(buffer_id), None).into());
        }

        if data_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize(data_size).into());
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset(buffer_offset).into());
        }
        if buffer_offset + data_size > size {
            return Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + data_size,
                buffer_size: size,
                side: CopySide::Destination,
            }
            .into());
        }

        let region = hal::command::BufferCopy {
            src: 0,
            dst: buffer_offset,
            size: data.len() as _,
        };

        // NOTE: submission_index is a lower bound approximation on the actual submission index,
        // since we relock mutably later on.
        let submission_index = {
            let (mut queue_inner_guard, _) = /*token1*/token.lock(&device.queue);
            /* // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we don't miss the scheduled writes.
            dst.life_guard
                .use_at(queue_inner_guard.active_submission_index + 1); */
            let pending_writes = &mut queue_inner_guard.pending_writes;
            let mut cmdbuf = device.post_prepare_stage_raw(pending_writes);
            unsafe {
                cmdbuf.pipeline_barrier(
                    super::all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    iter::once(hal::memory::Barrier::Buffer {
                        states: hal::buffer::Access::HOST_WRITE..hal::buffer::Access::TRANSFER_READ,
                        target: &stage.buffer,
                        range: hal::buffer::SubRange::WHOLE,
                        families: None,
                    })
                    .chain(transition.map(|pending| pending.into_hal(dst_raw))),
                );
                cmdbuf.copy_buffer(&stage.buffer, dst_raw, iter::once(region));
            };
            pending_writes.consume(stage, cmdbuf);
            pending_writes.dst_buffers.insert(buffer_id);
            queue_inner_guard.active_submission_index + 1
        };

        // Ensure the overwritten bytes are marked as initialized so they don't need to be nulled prior to mapping or binding.
        {
            /* drop((buffer_guard, token1)); */
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let dst = buffer_guard.get_mut(buffer_id).unwrap();
            // This is actually racy, but it was already racy before and I think it can't cause UB
            // (just incorrect behavior) if the user doesn't sensibly synchronize.
            dst.initialization_status
                .clear(buffer_offset..(buffer_offset + data_size));
            // NOTE: Since buffers is taken by at least read lock, we know the submission index is
            // not being updated by a call to queue_submit (which mutably locks hub.buffers).
            // Since we reject writes to currently mapped mapped-on-init buffers, and unmap
            // synchronizes with queue_write_buffer, we also know that the submission index is not
            // being updated by a call to queue_write_buffer.  Therefore, we can assume nobody else
            // is updating the submission index concurrently, so we just need to check whether the
            // current submission index is already higher than the one we want to use.
            let old_submission_index = dst.life_guard.submission_index.load(Ordering::Relaxed);
            if old_submission_index < submission_index {
                dst.life_guard.use_at(submission_index);
            }
        }

        Ok(())
    }

    /// Safety: must synchronize with any calls to texture_destroy or texture_drop for the same
    /// texture.
    ///
    /// NOTE: Currently NOT marked unsafe to make it easier to integrate with the gfx_select!
    /// macro, but it should be marked unsafe if this is backported to non-Veloren wgpu.
    pub fn queue_write_texture<'a, B: GfxBackend>(
        &'a self,
        queue_id: id::QueueId,
        destination: &ImageCopyTexture,
        data: &[u8],
        data_layout: &wgt::ImageDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_texture", "Queue");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = device_guard
            .get(queue_id)
            .map_err(|_| DeviceError::Invalid)?;
        let (texture_guard, mut token1) = hub.textures.read(&mut token);
        // TODO: Do most of this without the texture read lock taken.
        let (image_layers, image_range, image_offset) =
            texture_copy_view_to_hal(destination, size, &*texture_guard)?;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteTexture {
                to: destination.clone(),
                data: data_path,
                layout: *data_layout,
                size: *size,
            });
        }

        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            log::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let texture_format = texture_guard.get(destination.texture).unwrap().format;
        let bytes_per_block = conv::map_texture_format(texture_format, device.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        validate_linear_texture_data(
            data_layout,
            texture_format,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            bytes_per_block as wgt::BufferAddress,
            size,
            false,
        )?;

        let (block_width, block_height) = texture_format.describe().block_dimensions;
        let block_width = block_width as u32;
        let block_height = block_height as u32;

        if !conv::is_valid_copy_dst_texture_format(texture_format) {
            return Err(TransferError::CopyToForbiddenTextureFormat(texture_format).into());
        }
        let width_blocks = size.width / block_width;
        let height_blocks = size.height / block_width;

        let texel_rows_per_image = if let Some(rows_per_image) = data_layout.rows_per_image {
            rows_per_image.get()
        } else {
            // doesn't really matter because we need this only if we copy more than one layer, and then we validate for this being not None
            size.height
        };
        let block_rows_per_image = texel_rows_per_image / block_height;

        let bytes_per_row_alignment = get_lowest_common_denom(
            device.hal_limits.optimal_buffer_copy_pitch_alignment as u32,
            bytes_per_block,
        );
        let stage_bytes_per_row = align_to(bytes_per_block * width_blocks, bytes_per_row_alignment);

        let block_rows_in_copy =
            (size.depth_or_array_layers - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;
        let mut stage = device.prepare_stage(stage_size)?;

        let (mut trackers, token2) = token1.lock(&device.trackers);
        let (dst, transition) = trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                image_range,
                TextureUse::COPY_DST,
            )
            .unwrap();
        drop((trackers, token2));
        let &Texture {
            usage,
            aspects,
            dimension,
            kind,
            format: texture_format,
            ..
        } = dst;
        let &(ref dst_raw, _) = dst
            .raw
            .as_deref()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        // Safety: For the texture to be deallocated, it must be unregistered.  All methods that
        // unregister the texture, either require &mut on the device, or first require dropping
        // or destroying the texture, which can't happen until we return due to the precondition
        // on this function.
        let dst_raw: &'a B::Image = unsafe { mem::transmute(dst_raw) };
        drop((texture_guard, token1));

        if !usage.contains(wgt::TextureUsage::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }
        validate_texture_copy_range(
            destination,
            texture_format,
            kind,
            CopySide::Destination,
            size,
        )?;

        let bytes_per_row = if let Some(bytes_per_row) = data_layout.bytes_per_row {
            bytes_per_row.get()
        } else {
            width_blocks * bytes_per_block
        };

        let ptr = stage.memory.map(&device.raw, 0, stage_size)?;
        unsafe {
            profiling::scope!("copy");
            //TODO: https://github.com/zakarumych/gpu-alloc/issues/13
            if stage_bytes_per_row == bytes_per_row {
                // Fast path if the data isalready being aligned optimally.
                ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), stage_size as usize);
            } else {
                // Copy row by row into the optimal alignment.
                let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
                for layer in 0..size.depth_or_array_layers {
                    let rows_offset = layer * block_rows_per_image;
                    for row in 0..height_blocks {
                        ptr::copy_nonoverlapping(
                            data.as_ptr()
                                .offset((rows_offset + row) as isize * bytes_per_row as isize),
                            ptr.as_ptr().offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }
        stage.memory.unmap(&device.raw);
        if !stage.memory.is_coherent() {
            stage.memory.flush_range(&device.raw, 0, None)?;
        }

        // WebGPU uses the physical size of the texture for copies whereas vulkan uses
        // the virtual size. We have passed validation, so it's safe to use the
        // image extent data directly. We want the provided copy size to be no larger than
        // the virtual size.
        let max_image_extent = kind.level_extent(destination.mip_level as _);
        let image_extent = wgt::Extent3d {
            width: size.width.min(max_image_extent.width),
            height: size.height.min(max_image_extent.height),
            depth_or_array_layers: size.depth_or_array_layers,
        };

        let region = hal::command::BufferImageCopy {
            buffer_offset: 0,
            buffer_width: (stage_bytes_per_row / bytes_per_block) * block_width,
            buffer_height: texel_rows_per_image,
            image_layers,
            image_offset,
            image_extent: conv::map_extent(&image_extent, dimension),
        };
        // NOTE: submission_index is a lower bound approximation on the actual submission index,
        // since we relock mutably later on.
        let submission_index = {
            let (mut queue_inner_guard, _) = token.lock(&device.queue);
            /* // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we don't miss the scheduled writes.
            dst.life_guard
                .use_at(queue_inner_guard.active_submission_index + 1); */
            let pending_writes = &mut queue_inner_guard.pending_writes;
            let mut cmdbuf = device.post_prepare_stage_raw(pending_writes);
            unsafe {
                cmdbuf.pipeline_barrier(
                    super::all_image_stages() | hal::pso::PipelineStage::HOST
                        ..hal::pso::PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    iter::once(hal::memory::Barrier::Buffer {
                        states: hal::buffer::Access::HOST_WRITE..hal::buffer::Access::TRANSFER_READ,
                        target: &stage.buffer,
                        range: hal::buffer::SubRange::WHOLE,
                        families: None,
                    })
                    .chain(transition.map(|pending| pending.into_hal(dst_raw, aspects))),
                );
                cmdbuf.copy_buffer_to_image(
                    &stage.buffer,
                    dst_raw,
                    hal::image::Layout::TransferDstOptimal,
                    iter::once(region),
                );
            }
            pending_writes.consume(stage, cmdbuf);
            pending_writes.dst_textures.insert(destination.texture);
            queue_inner_guard.active_submission_index + 1
        };

        {
            let (texture_guard, _) = hub.textures.read(&mut token);
            let dst = texture_guard.get(destination.texture).unwrap();
            // NOTE: Since textures is taken by at least read lock, we know the submission index is
            // not being updated by a call to queue_submit (which mutably locks hub.textures).
            // Therefore, we can assume nobody else is updating the submission index concurrently,
            // so we just need to check whether the current submission index is already higher than
            // the one we want to use.
            let old_submission_index = dst.life_guard.submission_index.load(Ordering::Relaxed);
            if old_submission_index < submission_index {
                dst.life_guard.use_at(submission_index);
            }
        }

        Ok(())
    }

    pub fn queue_submit<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<(), QueueSubmitError> {
        profiling::scope!("submit", "Queue");

        let hub = B::hub(self);
        let mut token = Token::root();

        let callbacks = {
            let (mut temp_suspected, mut token) = token.lock(&hub.temp_suspected);
            temp_suspected.clear();
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = device_guard
                .get(queue_id)
                .map_err(|_| DeviceError::Invalid)?;

            // FIXME: I didn't bother to optimize the locks here, for the simple reason that they
            // are all (hopefully!) going to go away after the hubs refactor.  But if any are still
            // here, remember to try to drop as soon as possible to avoid blocking the queue.
            let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
            let (bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);
            let (mut buffer_guard, mut token) = hub.buffers.write(&mut token);
            let (texture_guard, mut token) = hub.textures.write(&mut token);
            let (mut trackers, mut token) = token.lock::<super::Trackers>(&device.trackers);
            let (mut queue_inner_guard, mut token) = token.lock(&device.queue);
            let super::QueueInner {
                ref mut pending_writes,
                raw: ref mut queue,
                ref mut active_submission_index,
            } = &mut *queue_inner_guard;

            let pending_write_command_buffer;
            pending_write_command_buffer = pending_writes.finish();
            for id in mem::take(&mut pending_writes.temp_buffer_resources) {
                let buffer = &mut buffer_guard[id];
                if buffer.raw.is_none() {
                    return Err(QueueSubmitError::DestroyedBuffer(id.0));
                }
                let old_submit_index = *buffer.life_guard.submission_index.get_mut();
                match buffer.map_state {
                    BufferMapState::Init { .. } if old_submit_index != 0 => {
                        let map_state = mem::replace(&mut buffer.map_state, BufferMapState::Idle);
                        if let BufferMapState::Init {
                            stage_buffer,
                            stage_memory,
                            ..
                        } = map_state
                        {
                            pending_writes
                                .temp_resources
                                .push((TempResource::Buffer(stage_buffer), stage_memory));
                        } else {
                            unreachable!();
                        }
                    }
                    _ => panic!("Buffer {:?} is still mapped", id),
                }
            }

            // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we synchronize with reads of the index from concurrent writers trying to add
            // new pending operations.
            let submit_index = active_submission_index.checked_add(1)
                // TODO: Abort?  Not clear just panicking is sound here (especially if we switch
                // this to an atomic).
                .expect("SubmissionIndex overflow: please try to avoid submitting to a queue more than\
                        u64::MAX times! (in the likely case that you did not actually do this, this is\
                        probably a bug in wgpu-core)");
            *active_submission_index = submit_index;

            let fence = {
                let mut signal_swapchain_semaphores = SmallVec::<[_; 1]>::new();
                let (mut command_buffer_guard, mut token) = hub.command_buffers.write(&mut token);

                if !command_buffer_ids.is_empty() {
                    profiling::scope!("prepare");

                    let (render_bundle_guard, mut token) = hub.render_bundles.read(&mut token);
                    let (_, mut token) = hub.pipeline_layouts.read(&mut token);
                    let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
                    let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
                    let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
                    let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
                    let (sampler_guard, _) = hub.samplers.read(&mut token);

                    let mut required_buffer_inits = RequiredBufferInits::default();

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
                        let cmdbuf = match command_buffer_guard.get_mut(cmb_id) {
                            Ok(cmdbuf) => cmdbuf,
                            Err(_) => continue,
                        };
                        #[cfg(feature = "trace")]
                        if let Some(ref trace) = device.trace {
                            trace.lock().add(Action::Submit(
                                submit_index,
                                cmdbuf.commands.take().unwrap(),
                            ));
                        }
                        if !cmdbuf.is_finished() {
                            continue;
                        }

                        required_buffer_inits
                            .add(&cmdbuf.buffer_memory_init_actions, &mut *buffer_guard)?;
                        // optimize the tracked states
                        cmdbuf.trackers.optimize();

                        for sc_id in cmdbuf.used_swap_chains.drain(..) {
                            let sc = &mut swap_chain_guard[sc_id.value];
                            if sc.acquired_view_id.is_none() {
                                return Err(QueueSubmitError::SwapChainOutputDropped);
                            }
                            if sc.active_submission_index != submit_index {
                                sc.active_submission_index = submit_index;
                                // Only add a signal if this is the first time for this swapchain
                                // to be used in the submission.
                                signal_swapchain_semaphores.push(sc_id.value);
                            }
                        }

                        // update submission IDs
                        for id in cmdbuf.trackers.buffers.used() {
                            // Check for map_on_init.
                            let buffer = &mut buffer_guard[id];
                            if buffer.raw.is_none() {
                                return Err(QueueSubmitError::DestroyedBuffer(id.0));
                            }
                            // let old_submit_index = *buffer.life_guard.submission_index.get_mut();
                            if !buffer.life_guard.use_at(submit_index) {
                                // println!("{:?}", buffer.map_state);
                                if let BufferMapState::Active { .. } = buffer.map_state {
                                    log::warn!("Dropped buffer has a pending mapping.");
                                    super::unmap_buffer(&device.raw, buffer)?;
                                    buffer.sync_mapped_writes = None;
                                    // NOTE: We do *not* dereference the actual B::Buffer mutably,
                                    // as it may be shared when we call here.
                                    let block = &mut buffer
                                        .raw
                                        .as_deref_mut()
                                        .expect("previously verified buffer.raw.is_some()")
                                        .1;
                                    // NOTE: Since the buffer is taken mutably, it definitely
                                    // synchronizes with any calls to map (it also synchronizes
                                    // with any calls to buffer_unmap_inner, since that function
                                    // takes the buffer at least immutably).  Moreover, the buffer
                                    // can never be mapped or unmapped after this, because there
                                    // are no more references to it.
                                    block.unmap(&device.raw);
                                }
                                temp_suspected.buffers.push(id);
                            } else {
                                match buffer.map_state {
                                    /* BufferMapState::Init { .. } if old_submit_index != 0 => {
                                        let map_state = mem::replace(
                                            &mut buffer.map_state,
                                            BufferMapState::Idle,
                                        );
                                        if let BufferMapState::Init {
                                            stage_buffer,
                                            stage_memory,
                                            ..
                                        } = map_state
                                        {
                                            pending_writes.temp_resources.push((
                                                TempResource::Buffer(stage_buffer),
                                                stage_memory,
                                            ));
                                        } else {
                                            unreachable!();
                                        }
                                    }, */
                                    BufferMapState::Idle => (),
                                    _ => panic!("Buffer {:?} is still mapped", id),
                                }
                            }
                        }
                        for id in cmdbuf.trackers.textures.used() {
                            let texture = &texture_guard[id];
                            if texture.raw.is_none() {
                                return Err(QueueSubmitError::DestroyedTexture(id.0));
                            }
                            if !texture.life_guard.use_at(submit_index) {
                                temp_suspected.textures.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.views.used() {
                            if !texture_view_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.texture_views.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bind_groups.used() {
                            if !bind_group_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.bind_groups.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.samplers.used() {
                            if !sampler_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.samplers.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.compute_pipes.used() {
                            if !compute_pipe_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.compute_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.render_pipes.used() {
                            if !render_pipe_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.render_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bundles.used() {
                            if !render_bundle_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.render_bundles.push(id);
                            }
                        }

                        // execute resource transitions
                        let mut transit = device.cmd_allocator.extend(cmdbuf);
                        unsafe {
                            // the last buffer was open, closing now
                            cmdbuf.raw.last_mut().unwrap().finish();
                            transit
                                .begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
                        }
                        log::trace!("Stitching command buffer {:?} before submission", cmb_id);
                        trackers.merge_extend_stateless(&cmdbuf.trackers);
                        CommandBuffer::insert_barriers(
                            &mut transit,
                            &mut *trackers,
                            &cmdbuf.trackers.buffers,
                            &cmdbuf.trackers.textures,
                            &*buffer_guard,
                            &*texture_guard,
                        );
                        unsafe {
                            transit.finish();
                        }
                        cmdbuf.raw.insert(0, transit);
                    }

                    log::trace!("Device after submission {}: {:#?}", submit_index, trackers);
                    if !required_buffer_inits.map.is_empty() {
                        super::Device::initialize_buffer_memory(
                            &mut *pending_writes,
                            required_buffer_inits,
                            &mut *trackers,
                            &mut *buffer_guard,
                            &device.cmd_allocator,
                        )?;
                    }
                }

                // now prepare the GPU submission
                let mut fence = device
                    .raw
                    .create_fence(false)
                    .or(Err(DeviceError::OutOfMemory))?;
                let signal_semaphores = signal_swapchain_semaphores
                    .into_iter()
                    .map(|sc_id| &swap_chain_guard[sc_id].semaphore);
                //Note: we could technically avoid the heap Vec here
                let mut command_buffers = Vec::new();
                // pending_write_command_buffer = pending_writes.finish();
                command_buffers.extend(pending_write_command_buffer.as_ref());
                for &cmd_buf_id in command_buffer_ids.iter() {
                    match command_buffer_guard.get(cmd_buf_id) {
                        Ok(cmd_buf) if cmd_buf.is_finished() => {
                            command_buffers.extend(cmd_buf.raw.iter());
                        }
                        _ => {}
                    }
                }

                unsafe {
                    queue.queues[0].submit(
                        command_buffers.into_iter(),
                        iter::empty(),
                        signal_semaphores,
                        Some(&mut fence),
                    );
                }
                fence
            };
            drop(swap_chain_guard);

            if let Some(comb_raw) = pending_write_command_buffer {
                device
                    .cmd_allocator
                    .after_submit_internal(comb_raw, submit_index);
            }

            // This will schedule destruction of all resources that are no longer needed
            // by the user but used in the command stream, among other things.
            let mut pending_write_resources =
                mem::take(&mut queue_inner_guard.pending_writes.temp_resources);
            let callbacks = match device.maintain(
                &hub,
                false,
                &*temp_suspected,
                bgl_guard,
                buffer_guard,
                texture_guard,
                trackers,
                queue_inner_guard,
                &mut token,
            ) {
                Ok(callbacks) => callbacks,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
            };
            drop(temp_suspected);

            profiling::scope!("cleanup");
            // this will register the new submission to the life time tracker
            token.lock(&device.life_tracker).0.track_submission(
                submit_index,
                fence,
                pending_write_resources.drain(..),
            );
            /* queue_inner_guard.pending_writes.temp_resources = pending_write_resources;
            drop(queue_inner_guard); */

            // return the command buffers to the allocator
            for &cmb_id in command_buffer_ids {
                if let (Some(cmd_buf), _) = hub.command_buffers.unregister(cmb_id, &mut token) {
                    device
                        .cmd_allocator
                        .after_submit(cmd_buf, &device.raw, submit_index);
                }
            }

            callbacks
        };

        // the map callbacks should execute with nothing locked!
        drop(token);
        super::fire_map_callbacks(callbacks);

        Ok(())
    }

    pub fn queue_get_timestamp_period<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
    ) -> Result<f32, InvalidQueue> {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        match device_guard.get(queue_id) {
            Ok(device) => {
                let (queue_inner_guard, _) = token.lock(&device.queue);
                Ok(queue_inner_guard.raw.queues[0].timestamp_period())
            }
            Err(_) => Err(InvalidQueue),
        }
    }
}

fn get_lowest_common_denom(a: u32, b: u32) -> u32 {
    let gcd = if a >= b {
        get_greatest_common_divisor(a, b)
    } else {
        get_greatest_common_divisor(b, a)
    };
    a * b / gcd
}

fn get_greatest_common_divisor(mut a: u32, mut b: u32) -> u32 {
    assert!(a >= b);
    loop {
        let c = a % b;
        if c == 0 {
            return b;
        } else {
            a = b;
            b = c;
        }
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    match value % alignment {
        0 => value,
        other => value - other + alignment,
    }
}

#[test]
fn test_lcd() {
    assert_eq!(get_lowest_common_denom(2, 2), 2);
    assert_eq!(get_lowest_common_denom(2, 3), 6);
    assert_eq!(get_lowest_common_denom(6, 4), 12);
}

#[test]
fn test_gcd() {
    assert_eq!(get_greatest_common_divisor(5, 1), 1);
    assert_eq!(get_greatest_common_divisor(4, 2), 2);
    assert_eq!(get_greatest_common_divisor(6, 4), 2);
    assert_eq!(get_greatest_common_divisor(7, 7), 7);
}
