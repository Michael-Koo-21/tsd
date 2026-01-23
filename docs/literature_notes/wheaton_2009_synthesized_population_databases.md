# Wheaton et al. (2009) — *Synthesized Population Databases: A US Geospatial Database for Agent-Based Models*

**What it is**
- RTI Press Methods Report (May 2009).
- Goal: build a *geospatially explicit synthetic population database* for agent-based models representing the US population in year 2000.

**Important scope note (relevance to our project)**
- This paper uses **decennial Census 2000-era sources** (incl. Census **PUMS** and **Summary File 3 (SF3)**), not the **ACS PUMS** person-file setup we’re using (ACS 2024 PUMS CA, adults).
- Still highly relevant as a “classic” synthetic population construction pipeline using PUMS microdata + aggregated controls.

**Data sources used**
- **TIGER** geographic boundary/network files.
- **Summary File 3 (SF3)** aggregated demographic variables (down to block group for full variable suite).
- **Public Use Microdata Sample (PUMS)** (described as a 5% sample of occupied/vacant housing units and people).
- A **crosswalk** mapping PUMAs to Census block group polygons.

**Synthesis approach (core method)**
- Two major steps:
  1) **Generate household locations** (point features) within each Census block group.
     - Number of points per block group equals SF3 household counts.
     - Locations random within block group, excluding water bodies.
  2) **Generate microdata records for 100% of households** by replicating/assigning PUMS household records.
     - Uses **iterative proportional fitting (IPF)**.
     - Implemented via Los Alamos **TRANSIMS Population Generator** (Beckman et al. procedure).
     - Concept: select/reweight PUMS households within a PUMA so that, within each block group, synthesized household characteristics match SF3 aggregated counts.

**Universe restriction / special populations**
- **Group quarters** are explicitly handled separately.
  - TRANSIMS Population Generator “does not include group quarters”; paper describes a separate procedure for group quarters residents.

**Which attributes they expect to fit best (i.e., what constraints they prioritize)**
- They expect best fit for:
  - number of people in household under age 18
  - household income
  - household size
  - household population
  - vehicles available

**Quality control**
- Compares aggregated synthesized attributes vs Census aggregated counts (example shown for Durham County, NC at block-group level).

**Implications for our ACS PUMS + CTGAN design fork**
- Reinforces the common “serious practice” pattern:
  - Apply **explicit universe restrictions** (they separate group quarters) rather than letting a model infer “not applicable” structure implicitly.
  - Use a **small, defensible set of control/constraint variables** (they call out a limited set where they expect best fit).
- It does *not* directly answer ACS-specific questions like PWGTP/replicate weights or ACS universe-coded missing values.

**How I would translate this into our pipeline**
- Prefer **filtering to the analysis universe early** (e.g., adults, in-universe records) over leaving lots of structural N/A categories.
- Keep the modeling variable set tight and interpretable; treat “special subpopulations” as separate strata if needed (analogous to group quarters handling), rather than forcing a single model to represent incompatible processes.
