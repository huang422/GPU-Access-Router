# Feature Specification: GPU Directer Toolkit

**Feature Branch**: `001-gpu-router-toolkit`
**Created**: 2026-03-11
**Status**: Draft
**Input**: User description: GPU Router toolkit for routing LLM compute to remote GPU server via Tailscale, with client/server roles, serial queue management, and pip-installable package

## Clarifications

### Session 2026-03-11

- Q: How should users override the default remote-first routing preference? → A: Config file sets the default routing mode (`remote`, `local`, or `auto`); a `prefer=` parameter on each API call allows per-call override without changing config.
- Q: What is the primary mechanism for viewing and editing configuration? → A: An editable plain-text config file (`~/.gpu-directer/config.toml`) plus CLI commands: `gpu-directer config show`, `config set key=value`, and `config edit` (opens in $EDITOR).
- Q: What should the server diagnostic command check and report? → A: Extended checks — Docker installed, Ollama container running, GPU passthrough working, Tailscale connected, available models listed, queue status — each with pass/fail result and a one-line fix hint on failure.
- Q: What should the router do when the requested model is missing on the remote server but exists locally (routing mode `auto`)? → A: Fall back to local Ollama and emit a warning message stating the remote server does not have the requested model.
- Q: What should the default per-request queue wait timeout be? → A: 300 seconds (5 minutes); users can override via config file or `prefer=timeout` per-call.

---

## Overview

GPU Directer is a modular, pip-installable Python toolkit that enables developers to use a remote GPU machine (gpu-server) from any laptop or cloud environment (gpu-client) over a private Tailscale network. The toolkit automatically detects the best available GPU source—preferring the remote GPU server—and provides a unified API so application code never needs to change. A serial request queue on the server ensures only one inference runs at a time, preventing GPU out-of-memory crashes.

**Two roles:**
- **gpu-server**: Ubuntu machine with NVIDIA GPU, running Ollama in Docker, reachable via Tailscale
- **gpu-client**: Any laptop or cloud instance (macOS/Ubuntu) that calls LLM APIs through the router

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Server Setup: Install and Configure GPU Server (Priority: P1)

A developer with a desktop GPU machine installs `gpu-directer[server]` and runs an interactive setup wizard. The wizard checks for Docker, pulls the Ollama Docker image, configures NVIDIA GPU passthrough, installs Tailscale, and prints the Tailscale IP address. After setup completes, the machine is ready to accept inference requests from clients.

**Why this priority**: Without a properly configured server, no client functionality is possible. This is the foundation of the entire system.

**Independent Test**: Can be fully tested on a single Ubuntu+NVIDIA machine by running the setup command and verifying that the Ollama API becomes accessible on the Tailscale network interface.

**Acceptance Scenarios**:

1. **Given** a fresh Ubuntu machine with an NVIDIA GPU and Docker installed, **When** the user runs `gpu-directer server setup`, **Then** Ollama Docker container starts with GPU access, Tailscale is installed and authenticated, and the server prints its Tailscale IP.
2. **Given** the server is running, **When** a client queries the server's health endpoint, **Then** the server responds with its status, available models, and current queue depth.
3. **Given** Docker is not installed, **When** the user runs `gpu-directer server setup`, **Then** the wizard detects the missing dependency, prints a clear installation instruction, and exits gracefully without partial setup.

---

### User Story 2 - Client Setup: Connect Client to Remote GPU Server (Priority: P1)

A developer on a laptop or Oracle Cloud instance installs `gpu-directer[client]` and runs an interactive setup wizard. The wizard asks for the server's Tailscale IP, tests connectivity, lists available Ollama models on the server, and saves the configuration. Going forward, any project on this machine can import `GPURouter` and call LLMs without specifying a host.

**Why this priority**: Client setup is the primary developer experience—it must be frictionless and verifiable before any project integration is useful.

**Independent Test**: Can be tested by installing the client package on a second machine connected to the same Tailscale network, running `gpu-directer client setup`, and confirming it discovers and lists the server's models.

**Acceptance Scenarios**:

1. **Given** Tailscale is installed and authenticated on both machines, **When** the user runs `gpu-directer client setup` and enters the server's Tailscale IP, **Then** the client verifies connectivity, lists all available Ollama models, and saves the configuration.
2. **Given** the server is unreachable (wrong IP or offline), **When** the user runs `gpu-directer client setup`, **Then** the wizard reports the connectivity failure with a diagnostic message and does not save a broken configuration.
3. **Given** setup is complete, **When** the user runs `gpu-directer client status`, **Then** the output shows the server address, current server status (online/offline), queue depth, and available models.

---

### User Story 3 - Unified API: Developer Uses GPURouter in Application Code (Priority: P1)

A developer adds `from gpu_directer import GPURouter` to their project and calls `router.chat(model, messages)`. The router automatically tries the remote GPU server first; if unavailable, it falls back to local Ollama. The developer's code never changes between environments.

**Why this priority**: The unified API is the core value proposition—it makes GPU resources transparent and portable across all environments.

**Independent Test**: Can be tested by writing a short script that calls `GPURouter().chat()` while the server is online (should route to server) and again while the server is offline (should route to local Ollama or raise a clear error).

**Acceptance Scenarios**:

1. **Given** the remote GPU server is reachable, **When** the developer calls `router.chat(model, messages)`, **Then** the request is routed to the remote server's Ollama instance and the response is returned in the same format as a local Ollama call.
2. **Given** the remote GPU server is offline, **When** the developer calls `router.chat(model, messages)`, **Then** the router falls back to local Ollama if available, transparently, without code changes.
3. **Given** neither remote server nor local Ollama is available, **When** the developer calls `router.chat(model, messages)`, **Then** a clear, actionable error is raised describing what was tried and why it failed.
4. **Given** the router is used as a drop-in replacement, **When** the developer switches between laptop and cloud environments, **Then** the same code works without modification because routing is determined by configuration, not code.

---

### User Story 4 - Serial Queue: Prevent Concurrent GPU Overload on Server (Priority: P2)

When multiple clients (or multiple projects on the same client) send inference requests simultaneously, the server queues them and processes them one at a time. Clients wait for their turn and receive a response with their estimated wait position. No request is lost, and the GPU never processes more than one inference at once.

**Why this priority**: Without queue management, concurrent requests can cause GPU out-of-memory crashes, making the server unusable. This is critical for multi-project or multi-user scenarios.

**Independent Test**: Can be tested by sending three simultaneous requests and verifying that all three eventually complete (not crash), that the server reports queue depth correctly, and that GPU memory never spikes to OOM.

**Acceptance Scenarios**:

1. **Given** the server is processing an inference, **When** a second client sends a request, **Then** the second request is queued and the client receives a queue position indicator while waiting.
2. **Given** multiple requests are queued, **When** the current inference completes, **Then** the next queued request starts immediately and the queue depth decreases by one.
3. **Given** a client has been waiting in queue for longer than the configured timeout (default 300 seconds), **When** the timeout is exceeded, **Then** the client receives a timeout error and the queue slot is released for other clients.

---

### User Story 5 - GitHub Install: Install from GitHub in Any Environment (Priority: P2)

A developer finds the project on GitHub and installs it with a single `pip install git+https://...` command. They can install just the client role, just the server role, or both. The package works on macOS and Ubuntu.

**Why this priority**: GitHub-based installation is the primary distribution mechanism for open-source adoption. It must be zero-friction.

**Independent Test**: Can be tested by running `pip install git+https://github.com/.../GPU-Directer-toolkit.git[client]` on a fresh macOS and Ubuntu environment and verifying the CLI tools and Python import work.

**Acceptance Scenarios**:

1. **Given** a machine with Python 3.8+ and pip, **When** the user runs `pip install git+https://...GPU-Directer-toolkit.git[client]`, **Then** the client tools and Python package install successfully with no manual dependency steps.
2. **Given** the package is installed, **When** the user runs `gpu-directer --help`, **Then** a clear help page with all available commands is shown.
3. **Given** a fresh Ubuntu server, **When** the user runs `pip install git+https://...GPU-Directer-toolkit.git[server]`, **Then** server setup tools are available and the install succeeds.

---

### User Story 6 - Documentation: Follow Setup Guide to Full Working System (Priority: P3)

A new user finds the project on GitHub, reads the README, follows the Tailscale setup guide, installs server and client components, and achieves a working end-to-end LLM call—all without prior knowledge of the project.

**Why this priority**: Open-source adoption depends entirely on documentation quality. A new user who cannot get started in 30 minutes will move on.

**Independent Test**: Can be tested via user testing: a person unfamiliar with the project follows only the README and achieves a working system within 30 minutes.

**Acceptance Scenarios**:

1. **Given** a new user with no prior context, **When** they follow the README Tailscale section, **Then** they can establish a Tailscale network between two machines by the end of the section.
2. **Given** a new user has completed Tailscale setup, **When** they follow the server setup section, **Then** the GPU server is running and accessible.
3. **Given** a new user has a running server, **When** they follow the client setup section and the quickstart code example, **Then** they successfully make their first LLM call through the router.

---

### Edge Cases

- What happens when the server's Tailscale IP changes after a reboot?
- How does the client handle network interruption mid-inference (lost Tailscale connection)?
- What happens when a requested model is not loaded on the server but exists as a pullable model?
- How does the queue behave when the server Docker container is restarted while requests are queued?
- What if the client's local Ollama has different models than the remote server? → Resolved: router falls back to local with a warning if the remote is missing the requested model.
- How does the system handle very long inference requests (large context windows) without timing out prematurely?
- What if multiple clients are configured to use the same server simultaneously from different geographic locations?

---

## Requirements *(mandatory)*

### Functional Requirements

**Server Role**

- **FR-001**: The server component MUST run Ollama inside a Docker container with NVIDIA GPU passthrough configured automatically during setup.
- **FR-002**: The server component MUST install and configure Tailscale during setup, producing a Tailscale IP address the user can share with clients.
- **FR-003**: The server MUST expose a health/status endpoint that reports current server status, available Ollama models, and queue depth.
- **FR-004**: The server MUST implement a serial request queue: only one inference request is processed at a time; all others wait in queue.
- **FR-005**: The server MUST enforce a configurable per-request queue wait timeout (default: 300 seconds), releasing the queue slot and returning an error to the client if the timeout is exceeded.
- **FR-006**: The server setup wizard MUST detect and report missing prerequisites (Docker, NVIDIA drivers, Tailscale) with actionable instructions before attempting configuration.
- **FR-006a**: The server MUST provide a `gpu-directer server doctor` diagnostic command that checks and reports pass/fail with a one-line fix hint for each of: Docker installed, Ollama container running, GPU passthrough active, Tailscale connected, at least one model available, queue depth.
- **FR-007**: The server MUST provide a command (`gpu-directer server models`) to list currently loaded Ollama models and their availability.

**Client Role**

- **FR-008**: The client component MUST provide an interactive setup wizard that accepts the server's Tailscale IP, verifies connectivity, and saves the configuration to `~/.gpu-directer/config.toml`.
- **FR-008a**: The toolkit MUST provide `gpu-directer config show` (display all settings), `gpu-directer config set <key>=<value>` (update a single setting), and `gpu-directer config edit` (open config file in $EDITOR) commands for both client and server roles.
- **FR-008b**: The config file MUST use a human-readable plain-text format (TOML) so users can directly open and edit it without CLI assistance.
- **FR-009**: During client setup, the wizard MUST query the server and display all available Ollama models to confirm the connection is working.
- **FR-010**: The client MUST provide a `GPURouter` class with a unified API that routes requests according to a configurable routing mode: `auto` (remote first, local fallback), `remote` (remote only), or `local` (local only). Default mode is `auto`.
- **FR-010a**: The routing mode MUST be settable in the configuration file and overridable per-call via a `prefer=` parameter (e.g., `router.chat(model, messages, prefer="local")`).
- **FR-011**: The `GPURouter` MUST support at minimum a `chat(model, messages)` method that returns responses compatible with the Ollama Python SDK format.
- **FR-012**: When routing mode is `auto` and the remote server is unreachable, the client MUST fall back to local Ollama automatically, without requiring code changes.
- **FR-012a**: When routing mode is `auto` and the remote server is reachable but does not have the requested model, the client MUST fall back to local Ollama (if the model exists locally) and emit a warning: "Model X not found on remote server, routing to local Ollama."
- **FR-013**: When neither remote server nor local Ollama is available, the client MUST raise a clear exception with a description of what was attempted.
- **FR-014**: The client MUST provide a status command that shows the configured server address, server reachability, current queue depth, and available models.

**Package & Distribution**

- **FR-015**: The toolkit MUST be installable via `pip install git+https://github.com/...` with role-specific extras: `[server]`, `[client]`, or `[all]`.
- **FR-016**: The package MUST provide a `gpu-directer` CLI entrypoint with sub-commands for setup, status, diagnostics, and config management for both roles. The full command surface MUST be documented in the README.
- **FR-017**: The package MUST support Python 3.8+ on Ubuntu and macOS.

**Documentation**

- **FR-018**: The project README MUST include a step-by-step Tailscale setup guide covering account creation, device enrollment, and IP discovery.
- **FR-019**: The project README MUST include separate quick-start sections for server and client setup, each completable in under 15 steps.
- **FR-020**: The project README MUST include a code example showing how to import and use `GPURouter` in a new project.

### Key Entities

- **GPUServer**: A configured Ubuntu+NVIDIA machine running Ollama Docker with Tailscale; identified by Tailscale IP; has a queue of pending inference requests.
- **GPUClient**: Any machine (laptop or cloud) with the client package installed; holds a configuration pointing to one GPUServer address.
- **GPURouter**: The Python object used in application code; holds routing logic, priority list, and fallback behavior; stateless per-call.
- **InferenceRequest**: A single LLM call with a model name, message list, and optional parameters; has a queue position, status (waiting/processing/complete/timeout), and response.
- **RequestQueue**: Server-side serial queue; enforces one-at-a-time processing; tracks position, timestamps, and timeout.
- **Configuration**: A TOML file at `~/.gpu-directer/config.toml` (per machine) storing Tailscale IPs, timeout values, routing mode (`auto`/`remote`/`local`), fallback preferences, and model preferences. Managed via `gpu-directer config` commands or direct file editing.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer with no prior knowledge of the project can install the server component, connect a client, and make a successful LLM call by following only the README, within 30 minutes.
- **SC-002**: The serial queue prevents GPU out-of-memory errors: when 5 simultaneous inference requests arrive, all 5 complete successfully and none cause a server crash.
- **SC-003**: Routing failover is transparent: when the remote server goes offline mid-session, the client's next request routes to local Ollama (if available) without the developer changing any code.
- **SC-004**: Installation succeeds on a fresh macOS or Ubuntu machine with Python 3.8+ using a single `pip install` command, with no additional manual steps required.
- **SC-005**: The server health endpoint responds in under 2 seconds under normal load, giving clients fast connectivity verification.
- **SC-006**: Client setup wizard completes (including connectivity verification and model listing) within 60 seconds on a typical Tailscale network.
- **SC-007**: The toolkit can be adopted in an existing project by adding fewer than 5 lines of code change (import, router instantiation, one API call).

---

## Assumptions

- The GPU server machine has an NVIDIA GPU with working drivers already installed before running setup.
- Docker is available or installable via standard package manager on the server.
- Tailscale account creation is handled manually by the user; the toolkit automates device enrollment and connection verification, not account sign-up.
- The server runs on Ubuntu 20.04 or later; client supports macOS 12+ and Ubuntu 20.04+.
- Internet connectivity is available on both machines during initial setup (for pulling Docker images, installing Tailscale, and installing the package from GitHub).
- The GPU server is connected via mobile hotspot (not a static IP), so Tailscale is the required network layer; no assumptions are made about the server's public IP.
- Only one GPU server is configured per client installation in v1; multi-server routing is a future enhancement.
- Ollama models must be pre-pulled on the server; the toolkit does not manage model downloads in v1.
- The primary use case is LLM text inference; image or multimodal models are supported if Ollama supports them, but no special handling is required in v1.

---

## Out of Scope (v1)

- Multi-server load balancing or failover between multiple GPU servers
- Automatic model pulling/downloading on the server triggered by client requests
- Web-based dashboard or GUI for monitoring queue status
- Authentication or authorization between clients and server (relies on Tailscale network-level trust)
- Windows support for either role
- Streaming inference responses (standard request-response only in v1)
