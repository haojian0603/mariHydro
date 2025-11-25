# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mariHydro is a Tauri-based desktop application for marine hydrological modeling using Shallow Water Equations. The project consists of two main components:
- **Core Library** (`swe_model`): Scientific computation library for hydrodynamic simulations
- **Tauri Application** (`marihydro`): Desktop frontend with database integration

## Development Commands

### Building the Project
```bash
# Build the core library
cargo build

# Build the Tauri application
cd src-tauri && cargo build

# Build for production
cd src-tauri && cargo tauri build
```

### Running the Application
```bash
# Run in development mode (with hot reload)
cd src-tauri && cargo tauri dev

# Run tests
cd src-tauri && cargo test
```

### Code Documentation
```bash
# Generate code collection report
python codelog.py
```

## Architecture Overview

### Core Engine Structure (`src-tauri/src/marihydro/`)

The physics engine is the heart of the application, implementing numerical schemes for Shallow Water Equations:

- **Physics Module** (`physics/engine.rs`): Contains the main computational engine with HLLC Riemann solver, MUSCL reconstruction, and conservation law implementations. This 22KB+ file is the core of the simulation.

- **Simulation Control** (`simulation/`): Orchestrates the numerical solver, manages computational grids, and handles I/O operations. The solver coordinates domain decomposition, time stepping, and convergence criteria.

- **Domain Management** (`domain/`): Handles mesh generation, boundary conditions, and geographic feature processing. Interfaces with GDAL for geospatial data processing.

- **Forcing Conditions** (`forcing/`): Manages external boundary conditions including wind fields, tidal forcing, and river inflows with temporal interpolation.

### Tauri Command Interface (`src-tauri/src/commands.rs`)

The frontend communicates with the Rust backend through these key commands:
- `get_default_config()`: Returns simulation configuration template
- `run_simulation()`: Executes hydrodynamic simulation with real-time progress reporting
- `save_simulation_record()`: Persists results to database

### Database Integration (`src-tauri/src/db.rs`)

Optional PostgreSQL/SQLite support for simulation record persistence. The application can run without database connectivity - all core functionality remains available.

### Geographic Processing (`src-tauri/src/marihydro/geo/`)

Implements coordinate reference system transformations using the PROJ library. Critical for handling different map projections and geographic datums in marine environments.

## Key Development Patterns

### Error Handling
The codebase uses structured error handling with `thiserror` for explicit error types. Physics errors, I/O errors, and configuration errors are distinctly categorized.

### Async Architecture
Simulation execution uses Tokio for non-blocking operation. Progress events are emitted to the frontend via Tauri's event system, allowing responsive UI during computation.

### Parallel Computing
Heavy computational loops leverage Rayon for data parallelism. The physics engine automatically parallelizes domain decomposition and flux computations across available CPU cores.

### Scientific Data Formats
NetCDF is the primary format for geospatial data interchange. The application reads/writes NetCDF files for bathymetry, boundary conditions, and simulation results.

## Testing Approach

Physics validation tests are located in `src-tauri/src/marihydro/physics/tests/`. Focus areas:
- Conservation law verification (mass, momentum conservation)
- Numerical scheme stability tests
- Boundary condition implementation validation

Run specific test modules:
```bash
cd src-tauri && cargo test --lib physics::tests::conservation
```

## Environment Configuration

### Database Setup (Optional)
Set `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgres://user:password@localhost/marihydro"
```

### Development Dependencies
- Rust stable toolchain
- Tauri prerequisites (platform-specific)
- GDAL development libraries (for geospatial processing)
- PROJ library (for cartographic projections)