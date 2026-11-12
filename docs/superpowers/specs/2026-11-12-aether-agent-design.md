# Design Document: Aether Agent (Autonomous Self-Improving Python Agent)

## 1. Project Identity
- **Nama Proyek:** `aether-agent` (Nama Paket Python: `aether_agent`, Perintah CLI: `aether`)
- **Deskripsi:** Proyek agen AI otonom sumber terbuka (*open-source*) berbasis Python modern yang memprogram ulang seluruh sistem arsitektur **Hermes Agent** (Nous Research). Proyek ini mencakup ekosistem lengkap: *TUI terminal*, *closed learning loop* (pembuatan skill otomatis berstandar `agentskills.io`), *multi-platform messaging gateway* (CLI, Telegram, Discord), *memory store* berbasis SQLite FTS5, *cron scheduler*, *subagent delegation*, dan *execution backends*, yang seluruhnya dikelola dengan Astral `uv`.

## 2. Intent & The Why
Hermes Agent dari Nous Research memperkenalkan standar baru bagi agen AI otonom yang tidak terikat pada satu IDE dan memiliki kemampuan belajar mandiri yang berkelanjutan (*closed learning loop*). Membangun proyek **`aether-agent`** yang memprogram ulang seluruh sistem ini secara bersih (*clean-room reimplementation*) bertujuan untuk:
- Memberikan basis kode yang elegan, modular, dan terstruktur rapi tanpa beban teknis lama (*legacy tech debt*).
- Menjamin portabilitas penuh di Windows (menggunakan PowerShell dan native process isolation) serta Linux/macOS.
- Membuka kebebasan penuh bagi developer untuk mengkustomisasi setiap subsistem (gateway, tools, skills, memory) dengan performa maksimal berbasis `uv` dan Python 3.11+.

## 3. Goals, Non-Goals, and Anti-Goals
### Goals
1. **Interactive Terminal TUI:**
   - Antarmuka terminal interaktif penuh (`prompt_toolkit` + `rich`).
   - Multiline prompt editor, syntax highlighting, slash-commands autocomplete (`/model`, `/skills`, `/gateway`, `/cron`, `/memory`, `/clear`, `/exit`).
   - Streaming token LLM real-time dan live spinner saat eksekusi tool.
2. **Autonomous ReAct Execution Engine:**
   - Siklus penalaran berkelanjutan (Plan -> Tool Call -> Observe -> Replan -> Synthesize).
   - Dukungan universal LLM: Cloud API (OpenAI, Gemini, OpenRouter, Anthropic) dan Local Model (Ollama/vLLM) via LiteLLM/OpenAI client standar.
3. **Multi-Backend Tool Runtime & Safety Guard:**
   - *Terminal Execution:* PowerShell di Windows, Bash di Unix/Linux.
   - *Safety Guard & Human-in-the-Loop:* Intersepsi otomatis perintah destruktif (misal: `rm -rf`, `format`, `del /f /s /q`) dengan prompt persetujuan pengguna.
   - *File Tools:* Operasi baca, tulis, patch/diff, dan pencarian direktori (*ripgrep/regex*).
   - *Code Execution Runner:* Eksekusi isolated Python script dengan timeout dan temporary sandbox.
   - *Web Retrieval:* Tool pencarian web dan ekstraksi konten halaman web.
4. **Closed Learning Loop & Skills Hub (`agentskills.io` Compliant):**
   - Refleksi pasca penyelesaian tugas kompleks: mendeteksi alur kerja yang berhasil dan dapat dipakai ulang.
   - Menyimpan skill dalam format direktori standar `skills/<skill-name>/SKILL.md` dengan YAML frontmatter (`name`, `description`) dan instruksi prosedural/kode.
   - Dynamic Skill Discovery: Mengindeks skill yang tersedia dan menginjeksikan deskripsinya secara efisien ke dalam system prompt agen.
5. **Memory & Long-Term Recall System:**
   - Database SQLite lokal dengan modul **FTS5 (Full-Text Search)** untuk pencarian instan riwayat percakapan lintas sesi.
   - Pemodelan profil preferensi pengguna (*user profiling* gaya Honcho) yang berkembang seiring waktu.
   - Context window manager dengan kompresi/ringkasan otomatis saat percakapan panjang.
6. **Multi-Platform Messaging Gateway:**
   - Layanan gateway asynchronous tunggal (`aether_agent.gateway`).
   - Adapter perpesanan: CLI (bawaan), Telegram Bot (`aiogram` / `python-telegram-bot`), dan Discord Bot (`discord.py`).
   - Sesi terpusat dan kontinuitas percakapan: pengguna dapat memulai tugas di CLI dan melanjutkannya dari Telegram.
7. **Scheduled Automations (Cron Scheduler):**
   - Penjadwalan tugas berulang (*scheduled jobs*) menggunakan ekspresi cron (`croniter`) atau interval waktu alami.
   - Eksekusi otomatis di latar belakang dengan pengiriman laporan hasil ke channel gateway yang ditentukan.
8. **Subagent Delegation:**
   - Kemampuan agen utama meluncurkan sub-agen terisolasi untuk tugas-tugas spesifik/paralel dengan batas kuota token dan toolset terbatas.
9. **Environment & Package Management:**
   - 100% didukung oleh Astral `uv` (`pyproject.toml`, virtual environment terisolasi, resolusi lockfile cepat).

### Non-Goals
- Tidak mewajibkan instalasi cluster Docker untuk penggunaan normal (Docker disediakan sebagai opsi execution backend lanjutan, bukan prasyarat instalasi).
- Tidak menyertakan frontend desktop Electron berat pada tahap awal; fokus utama adalah Terminal TUI kelas satu dan Gateway bots.

### Anti-Goals
- Menghindari framework monolitik "black box" (seperti LangChain berat) yang mengunci alur prompt dan tool parsing.
- Menghindari hardcoded provider API — pengguna bebas mengganti provider tanpa mengubah satu baris kode pun.
- Menghindari eksekusi perintah destruktif tanpa mekanisme approval.

## 4. Constraints
- **Bahasa & Runtime:** Python 3.11+
- **Platform:** Windows (prioritas utama pada workspace saat ini) dan POSIX (Linux/macOS).
- **Package Manager:** Astral `uv` (`uv venv`, `uv sync`, `uv run`).

## 5. Approaches Considered & Decision
1. **Minimalist Agent Core:** Hanya membangun loop dasar terminal. (*Ditolak karena tidak memenuhi kebutuhan pengguna yang menginginkan sistem lengkap seperti Hermes Agent*).
2. **Fork Codebase Resmi Langsung:** Menggunakan repo `NousResearch/hermes-agent` secara langsung. (*Ditolak karena pengguna menginginkan proyek baru yang memprogram ulang seluruh sistem dengan Hermes sebagai referensi*).
3. **Clean-Room Reimplementation of Full System (`aether-agent`) (*Dipilih*):** Membangun proyek baru dari awal dengan arsitektur modular yang mengimplementasikan seluruh pilar Hermes Agent (TUI, Tools, Learning Loop, Memory FTS5, Gateway, Cron, Subagents) menggunakan basis kode modern berbasis `uv`.

## 6. Architecture & System Modules
Struktur proyek `aether-agent`:
```
aether-agent/
├── pyproject.toml             # Konfigurasi dependensi uv & CLI script entrypoint
├── README.md
├── aether_agent/
│   ├── __init__.py
│   ├── config.py             # Konfigurasi terpusat (ENV, paths, model settings)
│   ├── core/
│   │   ├── agent.py          # ReAct Autonomous Agent Loop & Orchestrator
│   │   ├── llm.py            # Unified LLM Provider Adapter (Cloud & Ollama via LiteLLM)
│   │   ├── prompt.py         # System prompt, formatters, and dynamic context builder
│   │   ├── subagent.py       # Isolated subagent runner & delegator
│   │   └── types.py          # Data models (Message, ToolCall, ToolResult, SessionState)
│   ├── tools/
│   │   ├── registry.py       # Dynamic Tool Registry & JSON Schema Generator
│   │   ├── base.py           # BaseTool class & ApprovalPolicy interface
│   │   ├── terminal.py       # Shell command executor (PowerShell/Bash) dengan safety check
│   │   ├── files.py          # File operations (read, write, list, patch, search)
│   │   ├── runner.py         # Subprocess Python Code Execution Sandbox
│   │   ├── web.py            # Web search & web extract tool
│   │   └── approval.py       # Safety Guard rules, dangerous command patterns & prompt hook
│   ├── skills/
│   │   ├── engine.py         # Skills runtime & dynamic discovery loader
│   │   ├── generator.py      # Self-improving reflection & auto-skill writer
│   │   └── format.py         # agentskills.io standard parser & serializer (SKILL.md)
│   ├── memory/
│   │   ├── store.py          # SQLite database connection & migrations
│   │   ├── fts.py            # SQLite FTS5 Full-Text Search across sessions
│   │   ├── profile.py        # Long-term user preferences & knowledge store
│   │   └── context.py        # Sliding context window & conversation compressor
│   ├── gateway/
│   │   ├── server.py         # Central asynchronous messaging gateway daemon
│   │   ├── session_hub.py    # Cross-platform session continuity & routing
│   │   └── adapters/
│   │       ├── cli_adapter.py      # Terminal CLI input/output adapter
│   │       ├── telegram_adapter.py # Telegram bot adapter (aiogram)
│   │       └── discord_adapter.py  # Discord bot adapter (discord.py)
│   ├── cron/
│   │   ├── scheduler.py      # Async background cron engine (croniter)
│   │   └── task.py           # Scheduled job definition & delivery dispatcher
│   └── tui/
│       ├── app.py            # Full terminal interface entrypoint (`aether`)
│       ├── prompt.py         # Prompt-toolkit multiline input & slash autocomplete
│       └── views.py          # Rich formatting, status spinners, tool output streaming
├── skills/                   # Direktori skill bawaan dan hasil pembelajaran agen
│   └── coding-assistant/
│       └── SKILL.md
└── tests/                    # Test suite komprehensif (pytest)
    ├── test_agent_loop.py
    ├── test_tools.py
    ├── test_skills_engine.py
    ├── test_memory_fts.py
    └── test_gateway.py
```

## 7. Hermes Reference Alignment & Architectural Solutions
Berdasarkan tinjauan arsitektur dan penelusuran langsung pada codebase referensi `NousResearch/hermes-agent`, keputusan implementasi untuk 5 area krusial telah ditetapkan:

1. **Topology & Process Model (Dual-Mode Execution):**
   - **Mode CLI Interaktif (`aether`):** Menjalankan agen secara langsung *in-process* dengan TUI kaya fitur (`prompt_toolkit` + `rich`) untuk penggunaan interaktif developer.
   - **Mode Gateway Daemon (`aether gateway start`):** Menjalankan service background async yang mengelola adapter bot (Telegram, Discord) dan mesin penjadwal *Cron* 24/7.
   - **Shared State:** Keduanya membaca dan menulis ke database SQLite terpusat dengan mekanisme *Turn Lease* (kunci sesi) untuk menjamin kontinuitas tanpa konflik proses.

2. **Keamanan Multi-Platform & Kontrol Sesi Bersamaan:**
   - **Otorisasi Single-Tenant / Whitelist:** Akses bot Telegram & Discord dibatasi secara ketat berdasarkan ID Pengguna (`ALLOWED_USERS` di konfigurasi/env). Pesan dari ID yang tidak terdaftar akan otomatis ditolak guna melindungi terminal lokal dari eksekusi tidak sah.
   - **Turn Lease & Concurrency Guard:** Setiap sesi memiliki *mutex/lease* asynchronous (`turn_lease.py`). Jika permintaan baru tiba saat agen sedang memproses tugas, giliran baru dimasukkan ke antrean (*queue*) atau diberi notifikasi status sibuk.

3. **Approval Keamanan Lintas Saluran (CLI, Chat Bot, & Cron):**
   - **CLI:** Prompt konfirmasi interaktif standar (Yes/No).
   - **Telegram/Discord:** Tombol interaktif *inline keyboard* (Approve/Reject) dengan batas waktu tunggu (*timeout* default 60 detik).
   - **Cron / Unattended Execution:** Mode *fail-safe*; perintah berisiko tinggi yang terdeteksi otomatis ditolak dan dicatat ke log, kecuali task cron tersebut secara eksplisit memiliki izin otorisasi khusus.

4. **Isolasi Subagent & Integritas Memori FTS5:**
   - Subagent dieksekusi dengan *session_id* anak terisolasi (`ephemeral child session`).
   - Riwayat percakapan internal subagent tidak dimasukkan ke tabel `messages_fts` utama agar tidak mencemari pencarian percakapan pengguna; hanya ringkasan akhir (*synthesized result*) yang dikembalikan ke konteks agen induk.

5. **Tata Kelola Closed Learning Loop & Status Skill:**
   - Skill hasil pembelajaran mandiri disimpan ke direktori `skills/` dengan status metadata (`status: active` atau `status: draft`).
   - Perintah `/skills list`, `/skills inspect <name>`, dan `/skills remove <name>` disediakan di TUI dan Gateway agar pengguna memiliki visibilitas dan kontrol penuh atas skill yang dipelajari agen.

## 8. What's Left to the Builder
- Konfigurasi detail opsi CLI (`--model`, `--verbose`, `--gateway`, `--cron`).
- Penulisan unit tests menggunakan `pytest` dan `pytest-asyncio`.
- Template konfigurasi default `.env.example`.
- Penyiapan entrypoint script `pyproject.toml` sehingga dapat dijalankan dengan `uv run aether`.