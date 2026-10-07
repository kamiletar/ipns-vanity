# ipns-vanity

Перебор красивых (вэнити) IPNS-имён для ed25519-ключей libp2p. Имя имеет вид
`k51qzi5uqu5d` + 1 символ из `g…m` + остальное; подбирается слово либо в начале (позиция 12 или 13), либо в конце.

- `cpu/` — Rust: перебор на CPU (~0,5 млн ключей/с), а также проверка и запись ключей для GPU-версии.
- `gpu/` — CUDA: перебор на GPU (~33 млн ключей/с на RTX 5080 Laptop). Кандидатов всегда перепроверяет `cpu`.

## Сборка (Windows)

CUDA Toolkit 13.x и Visual Studio Build Tools (MSVC) нужны только для `gpu`.

```bat
cd cpu && cargo build --release
cd ..\gpu && "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" ^
  && nvcc -O3 -arch=sm_120 ipns_gpu.cu -o ipns_gpu.exe
```

`-arch=sm_120` — Blackwell (RTX 50xx); для другой карты поменяйте архитектуру.

## Использование

```bat
:: Самопроверка GPU против dalek (должно быть «не совпало: 0»)
ipns_gpu.exe --selftest 2000 | ..\cpu\target\release\ipns-vanity.exe check

:: Слово в конце имени ($), каталог для ключей, путь к cpu-утилите, [duty%] [пауза при °C]
ipns_gpu.exe "animatr$" C:\Users\<you>\ipns-keys ..\cpu\target\release\ipns-vanity.exe 100 86

:: Слово в начале: без $; можно несколько через запятую
ipns_gpu.exe "kami,letar" C:\Users\<you>\ipns-keys ..\cpu\target\release\ipns-vanity.exe

:: Только CPU
ipns-vanity.exe kami C:\Users\<you>\ipns-keys 12
```

Допустимы только `a-z` и `0-9`. Сложность растёт как 36ⁿ: 6 символов — около минуты на GPU, 7 — около 40 минут, 8 — около суток.

## Ключи

Файлы `*.key` — приватные ключи в формате libp2p-protobuf-cleartext
(`ipfs key import <имя> --format=libp2p-protobuf-cleartext <файл>`). Хранить вне репозитория, копии — только
в зашифрованном виде. В `.gitignore` они уже исключены.
