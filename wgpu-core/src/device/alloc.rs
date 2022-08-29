/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::DeviceError;
use crate::hub::{Mutex, MutexGuard};
use hal::device::Device as _;
use once_cell::sync::OnceCell;
use std::{borrow::Cow, cell::Cell, iter, ptr::NonNull};

pub(super) type Memory<B> = OnceCell<<B as hal::Backend>::Memory>;

#[derive(Debug)]
pub struct MemoryAllocator<B: hal::Backend> {
    allocator: Mutex<gpu_alloc::GpuAllocator<Memory<B>>>,
    properties: gpu_alloc::DeviceProperties<'static>,
}

#[derive(Debug)]
pub struct MemoryBlock<B: hal::Backend>(gpu_alloc::MemoryBlock<Memory<B>>);

pub(super) struct MemoryDevice<'a, B: hal::Backend> {
    device: &'a B::Device,
    alloc: Cell<Option<(u64, u32)>>,
    properties: &'a gpu_alloc::DeviceProperties<'static>,
}

impl<B: hal::Backend> MemoryAllocator<B> {
    pub fn new(mem_props: hal::adapter::MemoryProperties, limits: hal::Limits) -> Self {
        let mem_config = gpu_alloc::Config {
            dedicated_threshold: 32 << 20,
            preferred_dedicated_threshold: 8 << 20,
            transient_dedicated_threshold: 128 << 20,
            linear_chunk: 128 << 20,
            minimal_buddy_size: 1 << 10,
            initial_buddy_dedicated_size: 8 << 20,
        };
        let properties = gpu_alloc::DeviceProperties {
            memory_types: Cow::Owned(
                mem_props
                    .memory_types
                    .iter()
                    .map(|mt| gpu_alloc::MemoryType {
                        heap: mt.heap_index as u32,
                        props: gpu_alloc::MemoryPropertyFlags::from_bits_truncate(
                            mt.properties.bits() as u8,
                        ),
                    })
                    .collect::<Vec<_>>(),
            ),
            memory_heaps: Cow::Owned(
                mem_props
                    .memory_heaps
                    .iter()
                    .map(|mh| gpu_alloc::MemoryHeap { size: mh.size })
                    .collect::<Vec<_>>(),
            ),
            max_memory_allocation_count: if limits.max_memory_allocation_count == 0 {
                log::warn!("max_memory_allocation_count is not set by gfx-rs backend");
                !0
            } else {
                limits.max_memory_allocation_count.min(!0u32 as usize) as u32
            },
            max_memory_allocation_size: !0,
            non_coherent_atom_size: limits.non_coherent_atom_size as u64,
            buffer_device_address: false,
        };
        MemoryAllocator {
            allocator: Mutex::new(gpu_alloc::GpuAllocator::new(
                mem_config,
                gpu_alloc::DeviceProperties {
                    memory_types: properties.memory_types.clone(),
                    memory_heaps: properties.memory_heaps.clone(),
                    max_memory_allocation_count: properties.max_memory_allocation_count,
                    max_memory_allocation_size: properties.max_memory_allocation_size,
                    non_coherent_atom_size: properties.non_coherent_atom_size,
                    buffer_device_address: properties.buffer_device_address,
                },
            )),
            properties,
        }
    }

    pub(super) fn prepare<'a>(
        &'a self,
        device: &'a B::Device,
    ) -> (
        MutexGuard<'a, gpu_alloc::GpuAllocator<Memory<B>>>,
        MemoryDevice<'a, B>,
    ) {
        (
            self.allocator.lock(),
            MemoryDevice::<B> {
                properties: &self.properties,
                device,
                alloc: Cell::new(None),
            },
        )
    }

    pub(super) fn prepare_mut<'a>(
        &'a mut self,
        device: &'a B::Device,
    ) -> (
        &'a mut gpu_alloc::GpuAllocator<Memory<B>>,
        MemoryDevice<'a, B>,
    ) {
        (
            self.allocator.get_mut(),
            MemoryDevice::<B> {
                properties: &self.properties,
                device,
                alloc: Cell::new(None),
            },
        )
    }
}

impl<'a, B: hal::Backend> MemoryDevice<'a, B> {
    pub fn allocate(
        &self,
        mut allocator: MutexGuard<'a, gpu_alloc::GpuAllocator<Memory<B>>>,
        requirements: hal::memory::Requirements,
        usage: gpu_alloc::UsageFlags,
    ) -> Result<MemoryBlock<B>, DeviceError> {
        assert!(requirements.alignment.is_power_of_two());
        let request = gpu_alloc::Request {
            size: requirements.size,
            align_mask: requirements.alignment - 1,
            memory_types: requirements.type_mask,
            usage,
        };

        unsafe { allocator.alloc(self, request) }
            .map(MemoryBlock)
            .map_err(|err| match err {
                gpu_alloc::AllocationError::OutOfHostMemory
                | gpu_alloc::AllocationError::OutOfDeviceMemory => DeviceError::OutOfMemory,
                _ => panic!("Unable to allocate memory: {:?}", err),
            })
            .and_then(|block| {
                self.finish(allocator, block).map_err(|err| match err {
                    gpu_alloc::OutOfMemory::OutOfHostMemory
                    | gpu_alloc::OutOfMemory::OutOfDeviceMemory => DeviceError::OutOfMemory,
                })
            })
    }

    pub fn free(&self, allocator: &mut gpu_alloc::GpuAllocator<Memory<B>>, block: MemoryBlock<B>) {
        unsafe { allocator.dealloc(self, block.0) }
    }

    pub fn clear(&self, allocator: &mut gpu_alloc::GpuAllocator<Memory<B>>) {
        unsafe { allocator.cleanup(self) }
    }

    /// Finish a memory block allocation.  This must be called before doing anything else with the
    /// memory, as the rest of the code assumes it is already allocated.
    fn finish(
        &self,
        allocator: MutexGuard<'a, gpu_alloc::GpuAllocator<Memory<B>>>,
        block: MemoryBlock<B>,
    ) -> Result<MemoryBlock<B>, gpu_alloc::OutOfMemory> {
        // First, drop the mutex, so we don't block other, unrelated allocations.
        drop(allocator);
        // Find out whether an allocation just happened (we assume at most one allocation per lock).
        let alloc = self.alloc.replace(None);
        if let Some((size, memory_type)) = alloc {
            // println!("Deferred finish");
            unsafe {
                match self
                    .device
                    .allocate_memory(hal::MemoryTypeId(memory_type as _), size)
                {
                    Ok(memory) => {
                        // Initialize the OnceCell.  We are the only initializer, because anyone else with
                        // a reference to this block got it via reference to an earlier allocation.
                        block.0.memory().set(memory).unwrap();
                        Ok(block)
                    }
                    Err(_) => {
                        // On error, we just abort currently, since synchronizing deallocation
                        // with other threads would be a pain.  Long term, we should deallocate the
                        // memory on the GPU, update the block to tell clients that there's an
                        // error, and then abort.
                        log::error!("Out of device memory, aborting.");
                        std::process::abort();
                        /* Err(gpu_alloc::OutOfMemory::OutOfDeviceMemory) */
                    }
                }
            }
        } else {
            // println!("Eager finish");
            // Block waiting for the block to be initialized, so future calls to get succeed.
            block.0.memory().wait();
            Ok(block)
        }
    }
}

impl<B: hal::Backend> MemoryBlock<B> {
    pub fn bind_buffer(
        &self,
        device: &B::Device,
        buffer: &mut B::Buffer,
    ) -> Result<(), DeviceError> {
        let mem = self
            .0
            .memory()
            .get()
            .expect("Buffer was initialized when allocated.");
        unsafe {
            device
                .bind_buffer_memory(mem, self.0.offset(), buffer)
                .map_err(DeviceError::from_bind)
        }
    }

    pub fn bind_image(&self, device: &B::Device, image: &mut B::Image) -> Result<(), DeviceError> {
        let mem = self
            .0
            .memory()
            .get()
            .expect("Buffer was initialized when allocated.");
        unsafe {
            device
                .bind_image_memory(mem, self.0.offset(), image)
                .map_err(DeviceError::from_bind)
        }
    }

    pub fn is_coherent(&self) -> bool {
        self.0
            .props()
            .contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT)
    }

    pub fn map(
        &mut self,
        device: &B::Device,
        allocator: &MemoryAllocator<B>,
        inner_offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) -> Result<NonNull<u8>, DeviceError> {
        let offset = inner_offset;
        unsafe {
            self.0
                .map(
                    &MemoryDevice::<B> {
                        properties: &allocator.properties,
                        device,
                        alloc: Cell::new(None),
                    },
                    offset,
                    size as usize,
                )
                .map_err(DeviceError::from)
        }
    }

    pub fn unmap(&mut self, device: &B::Device, allocator: &MemoryAllocator<B>) {
        unsafe {
            self.0.unmap(&MemoryDevice::<B> {
                properties: &allocator.properties,
                device,
                alloc: Cell::new(None),
            })
        };
    }

    pub fn write_bytes(
        &mut self,
        device: &B::Device,
        allocator: &MemoryAllocator<B>,
        inner_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), DeviceError> {
        profiling::scope!("write_bytes");
        let offset = inner_offset;
        unsafe {
            self.0
                .write_bytes(
                    &MemoryDevice::<B> {
                        properties: &allocator.properties,
                        device,
                        alloc: Cell::new(None),
                    },
                    offset,
                    data,
                )
                .map_err(DeviceError::from)
        }
    }

    pub fn read_bytes(
        &mut self,
        device: &B::Device,
        allocator: &MemoryAllocator<B>,
        inner_offset: wgt::BufferAddress,
        data: &mut [u8],
    ) -> Result<(), DeviceError> {
        profiling::scope!("read_bytes");
        let offset = inner_offset;
        unsafe {
            self.0
                .read_bytes(
                    &MemoryDevice::<B> {
                        properties: &allocator.properties,
                        device,
                        alloc: Cell::new(None),
                    },
                    offset,
                    data,
                )
                .map_err(DeviceError::from)
        }
    }

    fn segment(
        &self,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> hal::memory::Segment {
        hal::memory::Segment {
            offset: self.0.offset() + inner_offset,
            size: size.or_else(|| Some(self.0.size())),
        }
    }

    pub fn flush_range(
        &self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> Result<(), DeviceError> {
        let segment = self.segment(inner_offset, size);
        let mem = self
            .0
            .memory()
            .get()
            .expect("Buffer was initialized when allocated.");
        unsafe {
            device
                .flush_mapped_memory_ranges(iter::once((mem, segment)))
                .or(Err(DeviceError::OutOfMemory))
        }
    }

    pub fn invalidate_range(
        &self,
        device: &B::Device,
        inner_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) -> Result<(), DeviceError> {
        let segment = self.segment(inner_offset, size);
        let mem = self
            .0
            .memory()
            .get()
            .expect("Buffer was initialized when allocated.");
        unsafe {
            device
                .invalidate_mapped_memory_ranges(iter::once((mem, segment)))
                .or(Err(DeviceError::OutOfMemory))
        }
    }
}

impl<B: hal::Backend> gpu_alloc::MemoryDevice<Memory<B>> for MemoryDevice<'_, B> {
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: gpu_alloc::AllocationFlags,
    ) -> Result<Memory<B>, gpu_alloc::OutOfMemory> {
        profiling::scope!("allocate_memory");

        assert!(flags.is_empty());

        let memory_props = /*gpu_alloc::MemoryPropertyFlags::from_bits_truncate(memory_type as u8)*/self.properties.memory_types[memory_type as usize].props;
        // println!("{:?}", memory_props);
        if memory_props.contains(gpu_alloc::MemoryPropertyFlags::HOST_VISIBLE) {
            // println!("Eager allocate_memory");
            self.device
                .allocate_memory(hal::MemoryTypeId(memory_type as _), size)
                .map(OnceCell::with_value)
                .map_err(|_| gpu_alloc::OutOfMemory::OutOfDeviceMemory)
        } else {
            // println!("Deferred allocate_memory");
            // NOTE: flags are guaranteed empty so we don't need to save them.
            self.alloc.set(Some((size, memory_type)));
            Ok(OnceCell::new())
        }
    }

    unsafe fn deallocate_memory(&self, memory: Memory<B>) {
        profiling::scope!("deallocate_memory");
        self.device.free_memory(
            memory
                .into_inner()
                .expect("Buffer was initialized when allocated."),
        );
    }

    unsafe fn map_memory(
        &self,
        memory: &mut Memory<B>,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, gpu_alloc::DeviceMapError> {
        profiling::scope!("map_memory");
        match self.device.map_memory(
            memory
                .get_mut()
                .expect("Buffer was initialized when allocated."),
            hal::memory::Segment {
                offset,
                size: Some(size),
            },
        ) {
            Ok(ptr) => Ok(NonNull::new(ptr).expect("Pointer to memory mapping must not be null")),
            Err(hal::device::MapError::OutOfMemory(_)) => {
                Err(gpu_alloc::DeviceMapError::OutOfDeviceMemory)
            }
            Err(hal::device::MapError::MappingFailed) => Err(gpu_alloc::DeviceMapError::MapFailed),
            Err(other) => panic!("Unexpected map error: {:?}", other),
        }
    }

    unsafe fn unmap_memory(&self, memory: &mut Memory<B>) {
        profiling::scope!("unmap_memory");
        self.device.unmap_memory(
            memory
                .get_mut()
                .expect("Buffer was initialized when allocated."),
        );
    }

    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, Memory<B>>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        profiling::scope!("invalidate_memory_ranges");
        self.device
            .invalidate_mapped_memory_ranges(ranges.iter().map(|r| {
                (
                    r.memory
                        .get()
                        .expect("Buffer was initialized when allocated."),
                    hal::memory::Segment {
                        offset: r.offset,
                        size: Some(r.size),
                    },
                )
            }))
            .map_err(|_| gpu_alloc::OutOfMemory::OutOfHostMemory)
    }

    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[gpu_alloc::MappedMemoryRange<'_, Memory<B>>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        profiling::scope!("flush_memory_ranges");
        self.device
            .flush_mapped_memory_ranges(ranges.iter().map(|r| {
                (
                    r.memory
                        .get()
                        .expect("Buffer was initialized when allocated."),
                    hal::memory::Segment {
                        offset: r.offset,
                        size: Some(r.size),
                    },
                )
            }))
            .map_err(|_| gpu_alloc::OutOfMemory::OutOfHostMemory)
    }
}
