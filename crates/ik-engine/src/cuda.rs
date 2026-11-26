//! Thin CUDA driver API layer: context, streams, events, memory, modules, launches and graphs.
//! kernite writes the kernels; this layer runs them with what kernite-cuda does not have (streams, graphs, async copies).

#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr::null_mut;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
    fn cuDeviceGet(d: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, d: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, d: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, d: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, n: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemHostAlloc(p: *mut *mut c_void, n: usize, flags: c_uint) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    fn cuMemHostGetDevicePointer_v2(d: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: CUstream) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: CUstream) -> CUresult;
    fn cuStreamCreate(s: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamCreateWithPriority(s: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult;
    fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult;
    fn cuStreamDestroy_v2(s: CUstream) -> CUresult;
    fn cuStreamSynchronize(s: CUstream) -> CUresult;
    fn cuStreamWaitEvent(s: CUstream, e: CUevent, flags: c_uint) -> CUresult;
    fn cuEventCreate(e: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventDestroy_v2(e: CUevent) -> CUresult;
    fn cuEventRecord(e: CUevent, s: CUstream) -> CUresult;
    fn cuEventSynchronize(e: CUevent) -> CUresult;
    fn cuEventQuery(e: CUevent) -> CUresult;
    fn cuEventElapsedTime(ms: *mut f32, a: CUevent, b: CUevent) -> CUresult;
    fn cuModuleLoadDataEx(m: *mut CUmodule, image: *const c_void, n: c_uint, opts: *mut c_int, vals: *mut *mut c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
        shared: c_uint, s: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuStreamBeginCapture_v2(s: CUstream, mode: c_int) -> CUresult;
    fn cuStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult;
    fn cuGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> CUresult;
    fn cuGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult;
    fn cuGraphExecDestroy(e: CUgraphExec) -> CUresult;
    fn cuGraphDestroy(g: CUgraph) -> CUresult;
    fn cuMemGetAllocationGranularity(g: *mut usize, prop: *const CUmemAllocationProp, option: c_int) -> CUresult;
    fn cuMemAddressReserve(p: *mut CUdeviceptr, size: usize, align: usize, addr: CUdeviceptr, flags: u64) -> CUresult;
    fn cuMemAddressFree(p: CUdeviceptr, size: usize) -> CUresult;
    fn cuMemCreate(h: *mut u64, size: usize, prop: *const CUmemAllocationProp, flags: u64) -> CUresult;
    fn cuMemRelease(h: u64) -> CUresult;
    fn cuMemMap(p: CUdeviceptr, size: usize, offset: usize, h: u64, flags: u64) -> CUresult;
    fn cuMemUnmap(p: CUdeviceptr, size: usize) -> CUresult;
    fn cuMemSetAccess(p: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) -> CUresult;
}

#[repr(C)]
struct CUmemLocation {
    ty: c_int, // CU_MEM_LOCATION_TYPE_DEVICE = 1
    id: c_int,
}

#[repr(C)]
struct CUmemAllocationProp {
    ty: c_int,           // CU_MEM_ALLOCATION_TYPE_PINNED = 1
    handle_types: c_int, // none
    location: CUmemLocation,
    win32: *mut c_void,
    flags: [u8; 8],
}

#[repr(C)]
struct CUmemAccessDesc {
    location: CUmemLocation,
    flags: c_int, // CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
}

pub type Result<T> = std::result::Result<T, String>;

pub fn check(r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorString writes a static string pointer.
    unsafe { cuGetErrorString(r, &mut s) };
    let msg = if s.is_null() { "?".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned() };
    Err(format!("{what}: CUDA error {r}: {msg}"))
}

macro_rules! ck {
    ($e:expr) => {
        check(unsafe { $e }, stringify!($e))
    };
}

/// The primary context of a device, current on the thread that made it.
pub struct Ctx {
    pub device: CUdevice,
    pub name: String,
    pub sm_count: u32,
    pub cc: (u32, u32),
}

impl Ctx {
    pub fn new(ordinal: i32) -> Result<Ctx> {
        ck!(cuInit(0))?;
        let mut device = 0;
        ck!(cuDeviceGet(&mut device, ordinal))?;
        let mut ctx = null_mut();
        ck!(cuDevicePrimaryCtxRetain(&mut ctx, device))?;
        ck!(cuCtxSetCurrent(ctx))?;
        let mut name = [0 as c_char; 256];
        ck!(cuDeviceGetName(name.as_mut_ptr(), 256, device))?;
        let attr = |a: c_int| -> Result<u32> {
            let mut v = 0;
            ck!(cuDeviceGetAttribute(&mut v, a, device))?;
            Ok(v as u32)
        };
        let sm_count = attr(16)?; // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        let cc = (attr(75)?, attr(76)?);
        Ok(Ctx { device, name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(), sm_count, cc })
    }

    pub fn sync(&self) -> Result<()> {
        ck!(cuCtxSynchronize())
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0, 0);
        ck!(cuMemGetInfo_v2(&mut f, &mut t))?;
        Ok((f, t))
    }
}

pub struct Stream(pub(crate) CUstream);
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// A stream that the synchronous copies (on the legacy default stream) are ordered with.
    pub fn new() -> Result<Stream> {
        let mut s = null_mut();
        ck!(cuStreamCreate(&mut s, 0))?;
        Ok(Stream(s))
    }
    /// A stream that runs apart from the legacy default stream: only async copies on it are ordered with it.
    pub fn new_nonblocking() -> Result<Stream> {
        let mut s = null_mut();
        ck!(cuStreamCreate(&mut s, 1))?; // CU_STREAM_NON_BLOCKING
        Ok(Stream(s))
    }
    /// new_nonblocking at the context's highest priority: once a block of another stream's kernel ends, a waiting block
    /// of this one's starts first.
    pub fn new_nonblocking_high() -> Result<Stream> {
        let (mut least, mut greatest) = (0, 0);
        ck!(cuCtxGetStreamPriorityRange(&mut least, &mut greatest))?;
        let mut s = null_mut();
        ck!(cuStreamCreateWithPriority(&mut s, 1, greatest))?;
        Ok(Stream(s))
    }
    pub fn sync(&self) -> Result<()> {
        ck!(cuStreamSynchronize(self.0))
    }
    /// The work queued on the stream from now on waits for the work the event was recorded after.
    pub fn wait(&self, e: &Event) -> Result<()> {
        ck!(cuStreamWaitEvent(self.0, e.0, 0))
    }
    pub fn begin_capture(&self) -> Result<()> {
        ck!(cuStreamBeginCapture_v2(self.0, 2)) // relaxed
    }
    pub fn end_capture(&self) -> Result<Graph> {
        let mut g = null_mut();
        ck!(cuStreamEndCapture(self.0, &mut g))?;
        let mut e = null_mut();
        let r = ck!(cuGraphInstantiateWithFlags(&mut e, g, 0));
        unsafe { cuGraphDestroy(g) };
        r?;
        Ok(Graph(e))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cuStreamDestroy_v2(self.0) };
    }
}

pub struct Event(CUevent);
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new(timing: bool) -> Result<Event> {
        let mut e = null_mut();
        ck!(cuEventCreate(&mut e, if timing { 0 } else { 2 }))?; // CU_EVENT_DISABLE_TIMING
        Ok(Event(e))
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        ck!(cuEventRecord(self.0, s.0))
    }
    pub fn sync(&self) -> Result<()> {
        ck!(cuEventSynchronize(self.0))
    }
    pub fn done(&self) -> bool {
        unsafe { cuEventQuery(self.0) == 0 }
    }
    pub fn ms_since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0.0;
        ck!(cuEventElapsedTime(&mut ms, start.0, self.0))?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cuEventDestroy_v2(self.0) };
    }
}

pub struct Graph(CUgraphExec);
unsafe impl Send for Graph {}

impl Graph {
    pub fn launch(&self, s: &Stream) -> Result<()> {
        ck!(cuGraphLaunch(self.0, s.0))
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { cuGraphExecDestroy(self.0) };
    }
}

/// Device memory, in bytes: its own allocation, or a view of memory that something else owns (a Region's).
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub len: usize,
    owned: bool,
}
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl DevBuf {
    pub fn new(len: usize) -> Result<DevBuf> {
        let mut ptr = 0;
        ck!(cuMemAlloc_v2(&mut ptr, len.max(1)))?;
        Ok(DevBuf { ptr, len, owned: true })
    }
    /// `len` bytes at `ptr`, which the view does not free.
    pub fn view(ptr: CUdeviceptr, len: usize) -> DevBuf {
        DevBuf { ptr, len, owned: false }
    }
    pub fn zeroed(len: usize, s: &Stream) -> Result<DevBuf> {
        let b = DevBuf::new(len)?;
        b.memset(0, s)?;
        Ok(b)
    }
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<DevBuf> {
        let b = DevBuf::new(size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.upload_at(0, data)
    }
    pub fn upload_at<T: Copy>(&self, off: usize, data: &[T]) -> Result<()> {
        let n = size_of_val(data);
        assert!(off + n <= self.len, "upload of {n} bytes at {off} past {}", self.len);
        ck!(cuMemcpyHtoD_v2(self.ptr + off as u64, data.as_ptr() as *const c_void, n))
    }
    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        self.download_at(0, n)
    }
    pub fn download_at<T: Copy + Default>(&self, off: usize, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        assert!(off + n * size_of::<T>() <= self.len);
        ck!(cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, self.ptr + off as u64, n * size_of::<T>()))?;
        Ok(v)
    }
    pub fn memset(&self, v: u8, s: &Stream) -> Result<()> {
        ck!(cuMemsetD8Async(self.ptr, v, self.len, s.0))
    }
    pub fn at(&self, off: usize) -> CUdeviceptr {
        assert!(off <= self.len);
        self.ptr + off as u64
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.owned {
            unsafe { cuMemFree_v2(self.ptr) };
        }
    }
}

/// Buffers at fixed addresses in one VmBuf that is mapped only at times: views handed out one after another, usable
/// while `on`. Prefill's buffers live in one, whose pages are the expert cache's while decode runs.
pub struct Region {
    vm: VmBuf,
    used: usize,
}

impl Region {
    /// Addresses for `len` bytes of views; no memory yet.
    pub fn reserve(len: usize, dev: CUdevice) -> Result<Region> {
        Ok(Region { vm: VmBuf::reserve(len, dev)?, used: 0 })
    }

    /// A view of `len` bytes after the last one, at a multiple of 256 bytes.
    pub fn take(&mut self, len: usize) -> DevBuf {
        let at = self.used.next_multiple_of(256);
        assert!(at + len <= self.vm.len, "region of {} bytes: {len} more at {at}", self.vm.len);
        self.used = at + len;
        DevBuf::view(self.vm.ptr + at as u64, len)
    }

    /// The memory that the views take when mapped: whole pages.
    pub fn bytes(&self) -> usize {
        self.vm.pages_for(self.used)
    }

    /// Forgets the views, to lay them out again; the region must have no memory.
    pub fn clear(&mut self) {
        assert_eq!(self.vm.mapped(), 0);
        self.used = 0;
    }

    /// Maps the views' pages (zeroed on `s`) or releases them. The caller makes sure no work in flight uses them.
    pub fn set(&mut self, on: bool, s: &Stream) -> Result<()> {
        self.vm.resize(if on { self.used } else { 0 }, s)
    }
}

/// Device memory at a fixed address range whose pages are backed on demand: the first `mapped()` bytes are usable.
/// Kernels and CUDA graphs keep the address while pages come and go, so memory can move between buffers. Pages from
/// `host_from` on are page-locked host memory at the same addresses: kernels read and write them over PCIe.
pub struct VmBuf {
    pub ptr: CUdeviceptr,
    /// the reserved bytes, a whole number of pages
    pub len: usize,
    pages: Vec<u64>,
    page: usize,
    dev: CUdevice,
    host_from: usize,
}
unsafe impl Send for VmBuf {}
unsafe impl Sync for VmBuf {}

impl VmBuf {
    fn prop(dev: CUdevice, host: bool) -> CUmemAllocationProp {
        // CU_MEM_LOCATION_TYPE_HOST_NUMA = 3, node 0
        let location = if host { CUmemLocation { ty: 3, id: 0 } } else { CUmemLocation { ty: 1, id: dev } };
        CUmemAllocationProp { ty: 1, handle_types: 0, location, win32: null_mut(), flags: [0; 8] }
    }

    /// The page size: the driver's minimum granularity (2 MiB on current GPUs).
    pub fn granularity(dev: CUdevice) -> Result<usize> {
        let mut g = 0;
        ck!(cuMemGetAllocationGranularity(&mut g, &Self::prop(dev, false), 0))?;
        Ok(g)
    }

    /// Whether pages can be host memory (CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED), and
    /// host pages are no larger than device pages.
    pub fn host_pages_supported(dev: CUdevice) -> bool {
        let mut v = 0;
        let mut g = usize::MAX;
        // SAFETY: plain queries
        let ok = unsafe { cuDeviceGetAttribute(&mut v, 141, dev) == 0 && v != 0 && cuMemGetAllocationGranularity(&mut g, &Self::prop(dev, true), 0) == 0 };
        ok && Self::granularity(dev).is_ok_and(|p| p % g == 0)
    }

    /// Reserves addresses for `len` bytes (rounded up to pages); no memory is mapped yet.
    pub fn reserve(len: usize, dev: CUdevice) -> Result<VmBuf> {
        let page = Self::granularity(dev)?;
        let len = len.max(1).div_ceil(page) * page;
        let mut ptr = 0;
        ck!(cuMemAddressReserve(&mut ptr, len, page, 0, 0))?;
        Ok(VmBuf { ptr, len, pages: vec![], page, dev, host_from: usize::MAX })
    }

    /// Pages mapped from now on at byte `bytes` (rounded up to pages) and past it are host memory.
    pub fn set_host_from(&mut self, bytes: usize) {
        self.host_from = self.pages_for(bytes) / self.page;
    }

    /// The mapped bytes in host memory.
    pub fn host_bytes(&self) -> usize {
        self.pages.len().saturating_sub(self.host_from) * self.page
    }

    /// The bytes that `bytes` of content take: whole pages.
    pub fn pages_for(&self, bytes: usize) -> usize {
        bytes.div_ceil(self.page) * self.page
    }

    pub fn mapped(&self) -> usize {
        self.pages.len() * self.page
    }

    /// Backs the first `bytes` (rounded up to pages) with memory, mapping or releasing pages at the end; new pages are
    /// zeroed on `s`. The caller makes sure no work in flight reads released pages.
    pub fn resize(&mut self, bytes: usize, s: &Stream) -> Result<()> {
        let want = self.pages_for(bytes).min(self.len) / self.page;
        let have = self.pages.len();
        if want < have {
            ck!(cuMemUnmap(self.ptr + (want * self.page) as u64, (have - want) * self.page))?;
            for h in self.pages.drain(want..) {
                ck!(cuMemRelease(h))?;
            }
        } else if want > have {
            for i in have..want {
                let mut h = 0;
                ck!(cuMemCreate(&mut h, self.page, &Self::prop(self.dev, i >= self.host_from), 0))?;
                if let Err(e) = ck!(cuMemMap(self.ptr + (i * self.page) as u64, self.page, 0, h, 0)) {
                    unsafe { cuMemRelease(h) };
                    return Err(e);
                }
                self.pages.push(h);
            }
            // the device reads and writes all pages; the host its own too
            let desc = [CUmemAccessDesc { location: CUmemLocation { ty: 1, id: self.dev }, flags: 3 }, CUmemAccessDesc { location: CUmemLocation { ty: 3, id: 0 }, flags: 3 }];
            let at = |i: usize| self.ptr + (i * self.page) as u64;
            let host = want.min(self.host_from).max(have);
            for (from, to, nd) in [(have, host, 1), (host, want, 2)] {
                if to > from {
                    ck!(cuMemSetAccess(at(from), (to - from) * self.page, desc.as_ptr(), nd))?;
                }
            }
            // zeroed: the new device pages at once, the host pages one at a time (a memset over two host pages, or
            // over a device page and a host one, is refused: invalid argument)
            if host > have {
                ck!(cuMemsetD8Async(at(have), 0, (host - have) * self.page, s.0))?;
            }
            for i in host..want {
                ck!(cuMemsetD8Async(at(i), 0, self.page, s.0))?;
            }
        }
        Ok(())
    }

    pub fn upload_at<T: Copy>(&self, off: usize, data: &[T]) -> Result<()> {
        let n = size_of_val(data);
        assert!(off + n <= self.mapped(), "upload of {n} bytes at {off} past {}", self.mapped());
        ck!(cuMemcpyHtoD_v2(self.ptr + off as u64, data.as_ptr() as *const c_void, n))
    }

    pub fn at(&self, off: usize) -> CUdeviceptr {
        assert!(off <= self.mapped());
        self.ptr + off as u64
    }
}

impl Drop for VmBuf {
    fn drop(&mut self) {
        unsafe {
            if !self.pages.is_empty() {
                cuMemUnmap(self.ptr, self.mapped());
            }
            for &h in &self.pages {
                cuMemRelease(h);
            }
            cuMemAddressFree(self.ptr, self.len);
        }
    }
}

pub fn copy_d2d(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: &Stream) -> Result<()> {
    ck!(cuMemcpyDtoDAsync_v2(dst, src, n, s.0))
}

pub fn copy_h2d_async(dst: CUdeviceptr, src: *const u8, n: usize, s: &Stream) -> Result<()> {
    ck!(cuMemcpyHtoDAsync_v2(dst, src as *const c_void, n, s.0))
}

pub fn copy_d2h_async(dst: *mut u8, src: CUdeviceptr, n: usize, s: &Stream) -> Result<()> {
    ck!(cuMemcpyDtoHAsync_v2(dst as *mut c_void, src, n, s.0))
}

/// Page-locked host memory, mapped into the device address space.
pub struct HostBuf {
    pub ptr: *mut u8,
    pub dev: CUdeviceptr,
    pub len: usize,
}
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl HostBuf {
    pub fn new(len: usize) -> Result<HostBuf> {
        let mut p = null_mut();
        ck!(cuMemHostAlloc(&mut p, len.max(1), 0x01 | 0x02))?; // PORTABLE | DEVICEMAP
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, len) };
        let mut dev = 0;
        ck!(cuMemHostGetDevicePointer_v2(&mut dev, p, 0))?;
        Ok(HostBuf { ptr: p as *mut u8, dev, len })
    }
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len / size_of::<T>()) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice<T: Copy>(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / size_of::<T>()) }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe { cuMemFreeHost(self.ptr as *mut c_void) };
    }
}

pub struct Module(CUmodule);
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

#[derive(Clone, Copy)]
pub struct Func(CUfunction);
unsafe impl Send for Func {}
unsafe impl Sync for Func {}

impl Module {
    pub fn load(ptx: &str) -> Result<Module> {
        let c = CString::new(ptx).map_err(|e| e.to_string())?;
        let mut m = null_mut();
        ck!(cuModuleLoadDataEx(&mut m, c.as_ptr() as *const c_void, 0, null_mut(), null_mut()))?;
        Ok(Module(m))
    }
    pub fn func(&self, name: &str) -> Result<Func> {
        let c = CString::new(name).unwrap();
        let mut f = null_mut();
        check(unsafe { cuModuleGetFunction(&mut f, self.0, c.as_ptr()) }, &format!("cuModuleGetFunction({name})"))?;
        Ok(Func(f))
    }
}

/// A kernel argument: its bytes, as the kernel's `.param` takes them.
pub trait Arg {
    fn ptr(&self) -> *mut c_void;
}
impl Arg for u32 {
    fn ptr(&self) -> *mut c_void {
        self as *const u32 as *mut c_void
    }
}
impl Arg for i32 {
    fn ptr(&self) -> *mut c_void {
        self as *const i32 as *mut c_void
    }
}
impl Arg for u64 {
    fn ptr(&self) -> *mut c_void {
        self as *const u64 as *mut c_void
    }
}
impl Arg for f32 {
    fn ptr(&self) -> *mut c_void {
        self as *const f32 as *mut c_void
    }
}

impl Func {
    /// Launches the kernel on `s`; `args` are the parameters in order.
    pub fn launch(&self, s: &Stream, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32, args: &[&dyn Arg]) -> Result<()> {
        let mut ps: Vec<*mut c_void> = args.iter().map(|a| a.ptr()).collect();
        ck!(cuLaunchKernel(self.0, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared, s.0, ps.as_mut_ptr(), null_mut()))
    }
}
