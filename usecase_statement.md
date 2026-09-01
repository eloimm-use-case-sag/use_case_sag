# Use Case Statement — Fragrance Shelf-Life Stability

## Context

The Fragrance Quality team lastly observed an unexpectedly reduced shelf-life stability on several test samples of a specific fragrance compound. Unable to pinpoint the root cause from their usual process controls, they requested sensor data from five manufacturing plants (Plant-01 through Plant-05).

A team of process engineers consolidated the data into a single dataset covering **90 days** of production across **10 synthesis reactors**, totalling approximately **259,000 batch records**.

## Domain knowledge: 

Shelf-life: It is the length of time that a commodity, here a fragrance, may be stored without becoming unfit for use, consumption, or sale.
Batch reactor production: The technicians introduce the reagents at the beginning of the cycle with valve systems, after the pressure and temperature conditions have been reached. A mixing system helps the reaction to happen homogenously. These different experimental conditions are influencing the cinetic of the reaction. Once finished, the batch reactor is emptied towards an optional separation step. 

## Dataset

*This is a synthetic dataset.*

**Table:** `fragrance_stability_dataset`

| Column | Category | Description | Normal Range |
| --- | --- | --- | --- |
| `batch_id` | Identifier | Unique production batch reference (format: BATCH-2026-XXXXXX) | — |
| `event_timestamp` | Identifier | Timestamp of the sensor reading | — |
| `reaction_temperature_celsius` | Process sensor | Temperature inside the synthesis reactor (°C) | 75–100 °C. Warm enough to drive the chemical reaction forward, but not so hot that the fragrance molecules break down or evaporate. |
| `vessel_pressure_bar` | Process sensor | Internal pressure of the reaction vessel (bar) | 2–3 bar. Slightly above normal atmospheric pressure (\~1 bar) to prevent volatile ingredients from evaporating during the reaction. Cannot be negative. |
| `refractive_index` | Process sensor | Optical refractive index of the mixture (dimensionless) | 1.43–1.49. Measures how light bends through the liquid — acts as a quick fingerprint to track whether the mixture composition is on target. Organic oils and fragrance compounds typically fall in this range. |
| `density_g_cm3` | Process sensor | Density of the fragrance compound obtained after a further separation step. (g/cm³) | 0.80–0.90 g/cm³. Fragrance oils are lighter than water (1.0 g/cm³). Density indicates purity — if it drifts outside this range, the separation step may not have worked correctly. |
| `ph_level` | Process sensor | Acidity/basicity of the reaction medium | 5.0–6.0. Slightly acidic (for reference: water is 7, lemon juice is \~2). This mild acidity is needed to catalyse the reaction while avoiding unwanted side reactions. |
| `mixing_torque_nm` | Process sensor | Torque applied by the agitator motor (N·m) | 9–12.5 N·m. Reflects how much effort the stirrer uses — related to the liquid's thickness (viscosity). Too high could mean the mixture is too viscous; too low could mean poor mixing. Cannot be negative. |
| `mass_flow_rate_kg_h` | Process sensor | Feed rate of raw materials into the reactor (kg/h) - As these are closed reactors it is a productivity indicator reflecting how fast they can be charged/emptied/purged | 380–520 kg/h. How fast materials flow in and out of the reactor through the piping system. Indicates operational throughput. Cannot be negative. |
| `plant_country` | Categorical | Country identifier of the manufacturing site | — |
| `production_line_id` | Categorical | Reactor identifier (Fragrance-Synthesis-Reactor-XX) | — |
| `stability_months` | Target | Measured shelf-life stability (continuous, 18–30 months) | 18–30 months |
| `purity_grade` | Derived | Quality tier derived from stability (Perfumery / Industrial / Technical) | — |
| `oxidation_risk_flag` | Derived | Risk indicator derived from stability (boolean) | — |


**Data quality note:** The engineers mentionned that sometimes the sensors fail to record correctly the measures. 

## Objectives

1. **Root Cause Analysis** — Identify which process parameters and conditions explain the anormal reduced shelf-life stability. Determine whether specific reactors, plants, or operating regimes are responsible.

2. **Predictive Model** — Build a model to predict `stability_months` from manufacturing sensor readings, enabling the Quality team to anticipate shelf-life at production time rather than waiting months for lab results. It must describes the confidence in the predictions and helps the team to explain the variations of stability. 

## Deliverables & Presentation

Your results will be presented to a mixed audience of **senior data scientists** (technical) and **non-technical business stakeholders**. 

You are free to choose the tools to implement your work and the format and medium on how to present it.