# Comprehensive Manual for Pain Recognition in Cattle and Calves
**Focus:** Visual Pain Signals, Facial Expressions (Grimace Scales), and Behavioral Assessments
**Target Audience:** Human Experts, Veterinarians, and Automated/LLM Vision Systems

---

## 1. State of the Art (SoTA) Non-Automated Methods
Pain assessment in bovine species (cattle, bulls, calves) is notoriously difficult due to their evolutionary adaptation as prey animals, which leads them to mask signs of distress. The current State of the Art (SoTA) relies on validated, structured observational tools used by veterinary experts. 

The most prominent SoTA methods include:
1. **UNESP-Botucatu Cattle Pain Scale (UCAPS):** The gold-standard behavioral scale for cattle, evaluating locomotion, interactive behavior, appetite, and miscellaneous behavioral indicators. It holds the highest "strength of evidence" in systematic reviews for construct and criterion validity. [@feighelstein-2025; q="UNESP-Botucatu Cattle Pain Scale (UCAPS), a validated body behavioral scoring system"] [@tomacheuski-2023; q="UCAPS [23], UPAPS [22] and USAPS [26] presented with overall ‘high’ strength of evidence"]
2. **The Cow Pain Scale (CPS):** A rapid, 7-point composite scale designed by Gleerup et al. (2015/2017) to be used practically on-farm. It combines facial expressions (the "Cow Pain Face") with posture and approach-response metrics.
3. **Bovine Grimace Scale (BGS) / Cow Pain Face:** Scales specifically isolating Facial Action Units (FAUs) in adult cattle.
4. **Calf Grimace Scale (CGS):** A recently developed 6-FAU scale specifically optimized for calves experiencing acute pain (e.g., castration, disbudding). [@farghal-2024; q="six facial action units (FAUs) were identified: ear position, orbital tightening"]

---

## 2. Core Facial Action Units (FAUs) / Grimace Scale Definitions

Grimace scales strictly evaluate the head and face. The standard scoring mechanism across most manual scales is a 3-point system: **0 = Not present** (normal), **1 = Moderately present**, **2 = Obviously present**.

### A. Ears
*   **Normal (0):** Ears directed forward, actively moving, or alert.
*   **Pain (1-2):** Ears held rigidly backward, asymmetric, or held very low ("lamb's ears"). 
*   *Adult specific:* "cows in pain keep their ears straight backwards or very low like lamb’s ears" [@gleerup-2017; q="cows in pain keep their ears straight backwards or very low"].
*   *Calf specific:* "One ear forward, one backward (front angle only)" scores 1; "both ears... directed caudally" scores 2 [@farghal-2024; q="Both ears backward (both angles): both ears"].

### B. Eyes (Orbital Tightening & Tension)
*   **Normal (0):** Eyes completely open, rounded, entire eyeball visible.
*   **Pain (1-2):** 
    *   *Orbital Tightening:* Eyelids close or squint. "Eyes obviously closed: eyelids are in contact with each other, the entire eyeball is not visible" [@farghal-2024; q="Eyes obviously closed: eyelids are in contact with each other"].
    *   *Tension Above the Eye:* Muscle contractions create visible furrows or a "beak" shape over the eye. "Tension of the muscles above the eyes may be seen as 'furrow lines'" [@gleerup-2017; q="Tension of the muscles above the eyes may be seen as 'furrow lines'"].
    *   *Other indicators:* Tense stare, "eye white showing, tension in the upper eyelid muscle... third eyelid protrusion" [@mota-rojas-2025; q="eye white showing, tension in the upper eyelid muscle"].

### C. Muzzle, Nostrils, and Lips
*   **Normal (0):** Relaxed skin, relaxed cashew/elliptical shaped nostrils, closed and relaxed mouth.
*   **Pain (1-2):**
    *   *Nostril Dilation:* Nostrils become enlarged, strained, appearing circular or heart-shaped. "nostrils orifices are obviously open and appear nearly circular or heart shape" [@farghal-2024; q="nostrils orifices are obviously open and appear nearly circular"].
    *   *Lip Tension:* Increased tonus of the lips.
    *   *Mouth Opening:* Jaws not in contact, tongue may appear. Highly indicative of acute pain/vocalizing in calves, but rare or transient [@farghal-2024; q="upper and lower jaws are not in contact with each other, the tongue may appear"].

### D. Cheek and Jaw (Masseter Muscle)
*   **Normal (0):** Skin over the cheek is relaxed, no visible lines.
*   **Pain (1-2):** 
    *   *Straining of chewing muscle:* "skin over the cheek area appears obviously strained and contraction of the cheek muscles causes appearance of clearly visible lines or deep grooves" [@farghal-2024; q="skin over the cheek area appears obviously strained and contraction"].

---

## 3. Composite & Behavioral Indicators (The Cow Pain Scale)

Because facial expressions can be subtle or fleeting, human experts combine FAUs with whole-body indicators. The **Cow Pain Scale** uses a 0–2 scoring system across 7 parameters [@gleerup-2017; q="The cow pain scale consists of 7 behaviours, evaluated from 0-2"]:

1.  **Attention towards surroundings:** (0) Active/attentive -> (2) Not attentive/withdrawn.
2.  **Head position:** (0) Head held high -> (2) Very low (below withers).
3.  **Ear position:** (0) Forward/actively moving -> (2) Lamb's ears / back.
4.  **Facial expression:** (0) Neutral -> (2) Tense expression (Pain Face).
5.  **Response to approach:** (0) Looks at observer -> (2) May not look, keeps head low, ears back, leaves slowly or shows avoidance.
6.  **Back position:** (0) Straight line -> (2) Arched back.
7.  **Lameness:** (0) Normal stride -> (2) Very lame, no support on one leg.

*Threshold Note:* A total score above 5 indicates the cow is likely in pain and requires intervention [@gleerup-2017; q="if the pain score is above 5, the cow could be in pain"].

---

## 4. Scoring Methodology & Human Expertise Benchmarks

Evaluating pain in cattle is highly subjective and depends heavily on the evaluator's training and empathy. 

*   **Inter-Observer Reliability (Human Baseline):**
    *   When human experts evaluate whole-body behavior (UCAPS), reliability is very high. "very good reliability for UCAPS scoring (ICC estimate of 0.881...)" [@feighelstein-2025; q="very good26 reliability for UCAPS scoring"].
    *   However, when human experts evaluate facial grimaces (BGS) on video, reliability drops significantly due to the subtlety of the expressions. "Observers reached moderate inter-observer reliability for BGS scoring (ICC estimate of 0.505...)" [@feighelstein-2025; q="Observers reached moderate inter-observer reliability for BGS scoring"].
*   **Machine vs. Human:** Deep learning vision models (e.g., YOLOv8-Pose + LSTM architectures tracking spatiotemporal micro-expressions) drastically outperform human veterinarians in consistency. "the machine significantly outperforms veterinarian using both BGS (AUC difference = 0.088, p<0.01) and UCAPS (AUC difference = 0.245, p<0.01)" [@feighelstein-2025; q="the machine significantly outperforms veterinarian using both BGS"]. A primary reason for this is temporal limits of human observation: "pain-related facial expressions in livestock often last between 0.3 and 0.7 seconds, rendering periodic manual observations insufficiently sensitive" [@patel-2025; q="pain-related facial expressions in livestock often last between 0.3 and 0.7 seconds, rendering periodic manual observations insufficiently sensitive"].

---

## 5. Methodological Differences & Physiological Variations

### A. Life Stages: Calves vs. Adults
*   **Facial Musculature:** Calves have fully developed facial muscles and display pain similarly to adults, but their behavioral responses may differ [@tschoner-2021; q="calves’ facial muscles may already be fully developed"].
*   **Mouth Opening:** Calves undergoing castration frequently show "mouth opening" due to acute vocalization, whereas this is less commonly used as an FAU in adult generic scales [@farghal-2024; q="mouth opening presented a change in 40, 15, and 10% of calves"].

### B. Breed and Phenotype Differences
*   **Morphology Masking Signals:** Certain breeds (e.g., Nelore/Bos indicus vs. Angus/Bos taurus) express FAUs differently or possess physical traits that hide them. For instance, the "pronounced chin" FAU was excluded from the Calf Grimace Scale because the presence of a dewlap in beef calves obscures the jawline [@farghal-2024; q="Pronounced chin was excluded as it was not possible to score due to the presence of the dewlap"].
*   **Color/Lighting:** Black fur (e.g., Angus) makes it much harder for humans and standard cameras to detect subtle muscle tension (furrow lines) without specialized lighting or contrast adjustments [@feighelstein-2025; q="color (white in Nelore bulls and black in Angus bulls), essential for diverse agricultural settings"].

### C. Sex Differences
*   **Pain Response:** Research suggests male and female cattle may have diverging ontogeny of pain processing, but currently, standard visual Grimace Scales are applied universally across sexes [@mclennan-2019; q="males have been shown to have a reduced sensitivity to pain in comparison with females as they age"].

---

## 6. Trick Situations, Confounders, and Edge-Cases

Automated systems and human observers must account for the following critical edge-cases that generate **False Positives** or **False Negatives**:

1.  **The Stoicism Factor (Prey Animal Masking):** "Cattle’s evolutionary inclination to mask pain, derived from predator-avoidance behaviors, severely complicates clinical assessments" [@patel-2025; q="Cattle’s evolutionary inclination to mask pain, derived from predator-avoidance behaviors, severely complicates clinical assessments"]. The mere presence of a human observer can cause a cow to suppress pain expressions. *Best Practice:* Use remote video monitoring or drones [@farghal-2024; q="habituation period: A drone equipped with a camera"].
2.  **Stress vs. Pain (False Positives):** Environmental stressors identical to husbandry practices (e.g., maternal separation, physical restraint in a tilt-table) trigger FAUs even without painful stimuli. "tension above the eye... had a high frequency of score 1... suggesting the moderate presence of this FAU during less aversive conditions rather than pain" [@farghal-2024; q="suggesting the moderate presence of this FAU during less aversive conditions rather than pain"]. 
    *   *Differentiator:* Ear position, orbital tightening, and nostril dilation are generally more specific to *pain*, while eye-tension can just mean *stress/fear*.
3.  **Anesthesia and Sedation:** Drugs like Isoflurane, Ketamine, or Xylazine directly alter facial muscle tonus, creating expressions that mimic pain. Grimace scales are highly unreliable while an animal is actively sedated/recovering from general anesthesia [@mota-rojas-2025; q="isoflurane increases the RGS score"].
4.  **Temporal Sparsity:** Pain expressions in cattle are not continuous; they are fleeting. "cows have evolved to minimize overt displays of pain, making such expressions brief, subtle, and often context-dependent" [@zhang-2025; q="making such expressions brief, subtle, and often context-dependent"]. Because expressions can be as brief as "0.3 and 0.7 seconds" [@patel-2025; q="pain-related facial expressions in livestock often last between 0.3 and 0.7 seconds"], analyzing single static frames is highly prone to error compared to tracking micro-expressions over a time-series (e.g., LSTM 5-frame windows).
5.  **Non-Painful Illness (Malaise/Fatigue):** A cow may display lowered head carriage, narrowed eyes, and drooped ears simply from exhaustion, metabolic illness, or post-operative fatigue, which can be misclassified as acute nociceptive pain [@mclennan-2019; q="other affective states such as fear or malaise also needs to be assessed, as these may interact with or obscure facial expression of pain"].
6.  **Environmental Artifacts (Lighting, Mud, and Thermal Sensors):** Physical and environmental obstructions generate false outputs in both humans and CV pipelines. "Farm-specific environmental factors—including inconsistent lighting conditions, occlusion by equipment or other animals, mud contamination, and motion blur—significantly degrade detection accuracy" [@patel-2025; q="Farm-specific environmental factors—including inconsistent lighting conditions, occlusion by equipment or other animals, mud contamination, and motion blur—significantly degrade detection accuracy"]. Additionally, if multimodal (thermal) imaging is used to complement visual scoring, "environmental factors including ambient temperature, humidity, and airflow significantly impact thermal measurement reliability" [@patel-2025; q="environmental factors including ambient temperature, humidity, and airflow significantly impact thermal measurement reliability"].