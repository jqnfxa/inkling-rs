//! Host->device bandwidth from pinned memory, alone and while CPU threads stream RAM: `ik-pcie`
use ik_engine::cuda::{Ctx, DevBuf, Event, HostBuf, Stream, copy_h2d_async};
fn main() -> Result<(), String> {
    let _ctx = Ctx::new(0)?;
    let s = Stream::new()?;
    let n = 256 << 20;
    let h = HostBuf::new(n)?;
    let d = DevBuf::new(n)?;
    let (e0, e1) = (Event::new(true)?, Event::new(true)?);
    let run = |label: &str| -> Result<(), String> {
        e0.record(&s)?;
        for _ in 0..8 {
            copy_h2d_async(d.ptr, h.ptr, n, &s)?;
        }
        e1.record(&s)?;
        e1.sync()?;
        let ms = e1.ms_since(&e0)? as f64;
        println!("{label}: {:.1} GB/s", 8.0 * n as f64 / ms / 1e6);
        Ok(())
    };
    run("pinned H2D alone")?;
    // CPU threads reading 2 GiB meanwhile
    let big: Vec<u64> = vec![1; 1 << 28];
    let stop = std::sync::atomic::AtomicBool::new(false);
    let bytes = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|sc| {
        for t in 0..8 {
            let (big, stop, bytes) = (&big, &stop, &bytes);
            sc.spawn(move || {
                let mut acc = 0u64;
                let part = big.len() / 8;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for v in &big[t * part..(t + 1) * part] {
                        acc = acc.wrapping_add(*v);
                    }
                    bytes.fetch_add((part * 8) as u64, std::sync::atomic::Ordering::Relaxed);
                }
                std::hint::black_box(acc);
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
        let b0 = bytes.load(std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        run("pinned H2D while 8 CPU threads read RAM").unwrap();
        let cpu = (bytes.load(std::sync::atomic::Ordering::Relaxed) - b0) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        println!("CPU threads meanwhile: {cpu:.1} GB/s");
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    Ok(())
}
