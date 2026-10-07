//! Перебор вэнити-IPNS имён (ed25519, libp2p-key, base36).
//! Использование: ipns-vanity <префикс> <каталог для ключей> [потоков]
//! Префикс — символы [a-z0-9], ищется сразу после общей части имени.

use ed25519_dalek::SigningKey;
use rand::{RngCore, rngs::OsRng};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ALPHABET: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Заголовок CIDv1 + libp2p-key + identity-multihash + protobuf публичного ключа ed25519.
const HEADER: [u8; 8] = [0x01, 0x72, 0x00, 0x24, 0x08, 0x01, 0x12, 0x20];

/// IPNS-имя: "k" + base36(HEADER || публичный ключ).
fn ipns_name(public: &[u8; 32]) -> String {
    let mut num = [0u8; 40];
    num[..8].copy_from_slice(&HEADER);
    num[8..].copy_from_slice(public);

    // Деление большого числа на 36 в столбик, цифры получаем с конца.
    let mut digits: Vec<u8> = Vec::with_capacity(64);
    let mut start = 0;
    while start < num.len() {
        let mut rem: u32 = 0;
        for b in num[start..].iter_mut() {
            let cur = (rem << 8) | *b as u32;
            *b = (cur / 36) as u8;
            rem = cur % 36;
        }
        digits.push(ALPHABET[rem as usize]);
        while start < num.len() && num[start] == 0 {
            start += 1;
        }
    }
    digits.push(b'k');
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

/// Закрытый ключ в формате libp2p-protobuf-cleartext (его принимает `ipfs key import`).
fn kubo_key_bytes(seed: &[u8; 32], public: &[u8; 32]) -> Vec<u8> {
    let mut out = vec![0x08, 0x01, 0x12, 0x40];
    out.extend_from_slice(seed);
    out.extend_from_slice(public);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Режимы для GPU-перебора: fromseed <seedhex> <каталог> <слово> и check (stdin: "seedhex pubhex").
    if args.get(1).map(String::as_str) == Some("fromseed") {
        let seed: Vec<u8> = (0..64).step_by(2).map(|i| u8::from_str_radix(&args[2][i..i + 2], 16).unwrap()).collect();
        let seed: [u8; 32] = seed.try_into().unwrap();
        let public = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
        let name = ipns_name(&public);
        // «слово$» — слово должно стоять в конце имени
        let ok = match args[4].strip_suffix('$') {
            Some(w) => name.ends_with(w),
            None => name[12..].starts_with(args[4].as_str()) || name[13..].starts_with(args[4].as_str()),
        };
        if !ok {
            eprintln!("кандидат не подтверждён: {name}");
            std::process::exit(1);
        }
        let dir = std::path::PathBuf::from(&args[3]);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{name}.key")), kubo_key_bytes(&seed, &public)).unwrap();
        println!("НАЙДЕНО: {name}");
        return;
    }
    if args.get(1).map(String::as_str) == Some("check") {
        use std::io::BufRead;
        let (mut ok, mut bad) = (0, 0);
        for line in std::io::stdin().lock().lines() {
            let line = line.unwrap();
            let mut it = line.split_whitespace();
            let (Some(s), Some(p)) = (it.next(), it.next()) else { continue };
            let seed: Vec<u8> = (0..64).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect();
            let seed: [u8; 32] = seed.try_into().unwrap();
            let public = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
            let want: String = public.iter().map(|b| format!("{b:02x}")).collect();
            if want == p { ok += 1 } else { bad += 1; if bad <= 3 { println!("расхождение: seed {s}\n  gpu {p}\n  cpu {want}"); } }
        }
        println!("совпало: {ok}, не совпало: {bad}");
        return;
    }
    if args.get(1).map(String::as_str) == Some("hist") {
        // Диагностика: какие символы бывают на позициях 12 и 13 имени.
        let mut h = [[0u32; 128]; 3];
        for _ in 0..200_000 {
            let n = ipns_name(&SigningKey::generate(&mut OsRng).verifying_key().to_bytes());
            for (k, pos) in [11usize, 12, 13].iter().enumerate() {
                h[k][n.as_bytes()[*pos] as usize] += 1;
            }
        }
        for (k, pos) in [11, 12, 13].iter().enumerate() {
            let seen: String = (0..128u8).filter(|c| h[k][*c as usize] > 0).map(|c| c as char).collect();
            println!("позиция {pos}: {seen}");
        }
        return;
    }
    if args.len() < 3 {
        eprintln!("ipns-vanity <префикс> <каталог> [потоков]");
        std::process::exit(2);
    }
    // Несколько слов через запятую: ищем все сразу, каждое — один раз.
    let wants: Vec<String> = args[1].to_lowercase().split(',').map(str::to_string).collect();
    assert!(
        wants.iter().all(|w| !w.is_empty() && w.bytes().all(|c| ALPHABET.contains(&c))),
        "в префиксах допустимы только a-z и 0-9"
    );
    let out_dir = std::path::PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out_dir).unwrap();
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    // Общая часть имени: смотрим на любом ключе, общий префикс у всех ed25519-имён один.
    let probe = ipns_name(&SigningKey::generate(&mut OsRng).verifying_key().to_bytes());
    let common = &probe[..12];
    let all_targets: Arc<Vec<String>> = Arc::new(wants.clone());
    let remaining = Arc::new(Mutex::new(all_targets.to_vec()));
    println!("общая часть: {common}, ищем: {wants:?}");

    let tries = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..threads {
        let (tries, stop, remaining, out_dir) = (tries.clone(), stop.clone(), remaining.clone(), out_dir.clone());
        let all_targets = all_targets.clone();
        handles.push(std::thread::spawn(move || {
            let mut seed = [0u8; 32];
            let mut local = 0u64;
            while !stop.load(Ordering::Relaxed) {
                OsRng.fill_bytes(&mut seed);
                let public = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
                let name = ipns_name(&public);
                local += 1;
                if local == 4096 {
                    tries.fetch_add(local, Ordering::Relaxed);
                    local = 0;
                }
                // Быстрая проверка без блокировки по неизменному списку; блокировка — только при совпадении.
                if all_targets.iter().any(|t| matches_word(&name, t)) {
                    let mut rem = remaining.lock().unwrap();
                    if let Some(i) = rem.iter().position(|t| matches_word(&name, t)) {
                        let path = out_dir.join(format!("{name}.key"));
                        std::fs::write(&path, kubo_key_bytes(&seed, &public)).unwrap();
                        println!("НАЙДЕНО: {name}\nключ: {}", path.display());
                        rem.remove(i);
                        if rem.is_empty() {
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }
        }));
    }

    // Печатаем скорость раз в 5 секунд.
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        let secs = started.elapsed().as_secs_f64();
        if (secs as u64) % 5 == 0 && secs > 1.0 {
            let n = tries.load(Ordering::Relaxed);
            println!("{:.0} ключей/с, всего {n}", n as f64 / secs);
            std::thread::sleep(Duration::from_millis(600));
        }
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// Слово стоит сразу после общей части (позиция 12) либо через один символ (позиция 13).
/// На позиции 12 бывают только g–m, поэтому слова на другие буквы возможны только со сдвигом.
fn matches_word(name: &str, word: &str) -> bool {
    name[12..].starts_with(word) || name[13..].starts_with(word)
}
