# Swine/Piglet Automatic Pain Recognition & Grimace Scale Manual

## 1. Scope & Purpose
This manual compiles State-of-the-Art (SoTA) methodologies, specific Facial Action Units (FAUs), vocalizations, and behavioral indicators for the holistic assessment of Affective States in swine (*Sus scrofa domesticus*). It is designed as a cheatsheet for human experts (veterinarians, researchers, stockpersons) and a rule-based guide for Large Language Models (LLMs) and Computer Vision (CV) systems automating welfare monitoring beyond basic pain detection.

---

## 2. State-of-the-Art (SoTA) Affective Assessment Methods
Assessment in swine relies on multimodal, non-invasive paradigms: **Composite Behavioral Scales** (assessing full-body posture), **Grimace Scales** (assessing facial muscle changes), and **Acoustic Analysis** (vocalizations).

### 2.1 The Gold Standard Behavioral Scale: UPAPS
The **UNESP-Botucatu Pig Composite Acute Pain Scale (UPAPS)** is currently the most robust behavioral assessment tool with a "high strength of evidence" [@tomacheuski-2023; q="UPAPS [22] presented a ‘high’ strength of evidence"]. It relies on 6 items:
1. Attention to Affected Area
2. Locomotion
3. Activity
4. Appetite
5. Interactive Behavior
6. Miscellaneous Behaviors
*Scoring*: A cut-off score of 6 out of 18 indicates the need for rescue analgesia [@tomacheuski-2023; q="18 (sum) Cut-off: 6 out of 18"].

### 2.2 The Grimace Scales (PGS & SFES)
There is no single universally accepted Grimace Scale for swine; rather, different scales have been validated based on the methodology and life stage:
*   **PGS-B (Viscardi et al. Method):** The most practical pen-side tool for piglets, utilizing 3 core FAUs (Orbital Tightening, Ear Position, Cheek Tightening/Nose Bulge) [@viscardi-2017; q="Piglet Grimace Scale-b. PGS-B: Piglet Grimace Scale-a"].
*   **PGS-A (Di Giminiani et al. Method):** A highly granular scale initially testing 10 FAUs, refined to 7, but ultimately finding that *only* orbital tightening significantly changed post-procedure [@di-giminiani-2016; q="orbital tightening was the only FAU to significantly change"].
*   **SFES (Sow Facial Expression Scale):** Developed for adult females (sows), focusing on 5 FAUs (Tension Above Eyes, Snout Angle, Neck Tension, Temporal Tension, Ear Position) [@tomacheuski-2023; q="SFES: Sow Facial Expression Scale"].

### 2.3 Acoustic/Vocalization Assessment
Vocal communication in pigs is highly developed and correlates strongly with affective states. Calls are categorized into low-frequency (LF; grunts) and high-frequency (HF; screams/squeals).
*   **Acoustic Features:** Call duration, amplitude modulation rate, pitch, and intensity reliably encode valence and arousal across both LF and HF calls.

---

## 3. Holistic Affective State Indicators (Visual, Behavioral & Vocal)

### 3.1 Negative_Nociceptive (Pain)
*   **Facial:** Pronounced orbital tightening (squinting), cheek tension, and nose bulge.
*   **Postural:** Increased transition from lateral to sternal recumbency to guard painful areas [@vullo-2020; q="increase in lying in sternal position compared to lateral position observed after surgery may be interpreted in the same way"]. "Tail jamming" or tucked low tail posture is also highly indicative of localized pain like tail-docking [@di-giminiani-2016; q="increase in “middle,” “low,” and “tucked low” and a decrease in the “high” tail position"].
*   **Behavioral:** Isolation or conversely, strong social cohesion/huddling as a protective mechanism. Decreased general locomotion and activity.
*   **Vocal:** Painful procedures elicit vocalizations characterized by higher frequency and intensity. Specifically, the metrics of pitch, amplitude, and intensity increase linearly with pain severity [@cordeiro-2018; q="Pitch frequency, maximum amplitude, and intensity presented similar behavior, tending to increase successively from pain-free to marking, and after tail trimming, and castration"].

### 3.2 Negative_Affective (Stress/Fear)
*   **Facial:** Visually indistinguishable from physical pain in piglets regarding orbital tightening, driven purely by fear and handling restraint [@di-giminiani-2016; q="restraint of the piglet while images were taken is likely to have induced even greater stress"].
*   **Behavioral:** Active escape attempts, freezing, or hyper-vigilance.
*   **Vocal:** Increased call duration and amplitude modulations compared to positive contexts.

### 3.3 Positive_Welfare (Joy/Enrichment)
*   **Behavioral:** Play, synchronous activity, foraging, relaxed lateral lying without defensive grouping.
*   **Vocal:** Acoustic markers show clear separation from negative states. Calls in positive scenarios are shorter and exhibit fewer amplitude modulations [@briefer-2022; q="positive calls, whether they are LF or HF, are shorter and contain less amplitude modulations than negative calls"].

### 3.4 Neutral_Resting (Baseline)
*   **Behavioral/Postural:** Lying laterally (when thermoregulation allows), absence of tucked tails.
*   **Facial:** Ears relaxed and forward (in breeds where applicable), eyes open fully (if awake) or closed naturally without surrounding skin tension.

---

## 4. Core Facial Action Units (FAUs) Dictionary

### 4.1 Orbital Tightening (Eye Squeeze)
*The most reliable and evolutionarily conserved indicator of pain across species.*
*   **Definition:** The narrowing of the orbital area as the eyelids are squeezed together.
*   **Viscardi (PGS-B) Scoring:** 2-point scale. `0 = Absent` (eyes wide open), `1 = Present` (eyes narrowed or closed) [@viscardi-2017; q="orbital tightening is scored on a 2-point scale (0 = not present; 1 = present)"].
*   **Di Giminiani (PGS-A) Scoring:** 3-point scale. `0 = Not present`, `1 = Moderately present` (eyes halfway closed/teardrop shape), `2 = Obviously present` (eyes closed more than halfway/fully tightened) [@di-giminiani-2016; q="orbital tightening was placed on a 3-point scale"].
*   **Reliability:** High. ICC = 0.90 to 0.95 in expert panels [@viscardi-2017; q="0.90 for orbital tightening"] [@di-giminiani-2016; q="0.95 among 30 raters for orbital tightening"].

### 4.2 Cheek Tightening / Nose Bulge
*Note: In rodents, "nose bulge" refers to a protuberance on the bridge of the nose. In pigs, it refers to skin folds above the snout.*
*   **Definition:** The appearance of a bulge of skin on the snout, accompanied by visible tension/indentations in the cheeks. 
*   **Scoring (0-2):** `0 = Absent` (smooth snout, relaxed cheeks); `1 = Moderately present` (bulge is apparent, slight cheek indentations); `2 = Obviously present` (largely bulging snout with significant skin folds, obvious cheek indentations) [@viscardi-2017; q="nose bulge refers to several skin folds above the snout"].
*   **Reliability:** Moderate to Excellent. ICC = 0.54 to 0.81 depending on the study [@viscardi-2017; q="0.81 for cheek tightening/nose bulge"].

### 4.3 Ear Position
*   **Definition:** The ears are drawn back from their forward-facing, relaxed position. 
*   **Scoring (0-2):** `0 = Absent` (ears kept forward); `1 = Moderately present` (ears slightly moved backwards); `2 = Obviously present` (ears drawn heavily back or folded/curled) [@viscardi-2017; q="ears are drawn back from forward (baseline) position"].
*   **Reliability & Limitations:** Moderate to Good (ICC = 0.67 to 0.73) [@viscardi-2017; q="0.73 for ear position"]. However, as pigs grow, muscle prominence changes, reducing ear mobility. This makes ear position unreliable in growing or adult pigs [@vullo-2020; q="muscles pertaining to the scutiform cartilage in pigs, such as the parotid-auricular muscle, could in time become less prominent and reduce the possibility of ear movement"].

### 4.4 Temporal Tension & Lip Contraction (Di Giminiani / Lou Methods)
*   **Temporal Tension:** Skin over the temporal muscles appears tightly drawn. Difficult to score consistently (ICC = 0.40) [@di-giminiani-2016; q="temporal tension (0.40), lip contraction (0.28)"].
*   **Lip Contraction:** The canine notch becomes pronounced, and the upper lip pulls apart. Excluded from many operational scales because it cannot be seen in front-profile images [@di-giminiani-2016; q="lip contraction... was ultimately excluded as it could not be scored in 30% of images"].

---

## 5. Edge-Cases, Trick Situations, & Confounders
When programming LLMs or training CV models, the following physiological and environmental constraints must be heavily weighted:

### A. Life-Stage Reversal (Piglets vs. Adult Sows)
Pain manifestations in the eyes reverse depending on the age of the pig:
*   **Piglets:** Pain = **Eyes closed** or tightened [@viscardi-2017; q="piglets, eyes completely or partially closed and fully tightened are considered indicative of pain"].
*   **Adult Sows:** Severe Pain = **Eyes completely wide open** with ocular sclera (white of the eye) highly visible [@mota-rojas-2025; q="in sows, eyes completely open with the ocular sclera visible is an indication of severe pain (2 points)"].

### B. Breed Morphology & Ear Position
Fast-growing commercial breeds and older pigs experience morphological changes that nullify "Ear Position" as a reliable metric:
*   **Issue:** Rapid muscle growth, adipose tissue accumulation, and thick dermis reduce ear mobility. 
*   **Floppy Ears:** In some breeds, the ear pinnae fold under their own weight, concealing the eyes and making "ears drawn back" impossible to identify [@lou-2022; q="rapid growth then changes the morphology of the face, some breeds seeing their ear pinnae fold under their own weight"].

### C. Neonatal Confounders (<24 hours old)
*   **Issue:** Piglets less than 24 hours old have rapid morphological shifts, wet ears, and natural, excessive facial wrinkling. This baseline wrinkling triggers false positives for "Cheek Tightening/Nose Bulge" [@lou-2022; q="newly born pigs have wet ears and excessive face wrinkling that would otherwise settle with time"].

### D. Stress vs. Pain (The Handling Confound)
*   **Issue:** Fear and restraint stress trigger identical FAUs (especially orbital tightening) to pain. 
*   **Observation:** In studies where piglets were physically restrained for photos (e.g., during castration), grimace scales failed to detect pain differences because the restraint stress maxed out the facial expressions in both control and pain groups [@di-giminiani-2016; q="restraint of the piglet while images were taken is likely to have induced even greater stress"]. *Automated systems MUST use unhandled, freely moving footage.*

### E. Thermoregulation vs. Pain (Behavioral Confound)
*   **Issue:** Group huddling and sternal recumbency are recognized protective behaviors for pain. However, they are also primary thermoregulatory responses to cold environments.
*   **Observation:** When evaluating social cohesion as a pain marker, ambient temperature must be verified to prevent false positives [@vullo-2020; q="Another possible reason for lying in contact with conspecifics could be thermoregulation; it has been reported that lying behavior is an important thermoregulatory mechanism in pigs. When the temperature lowers, pigs adapt their behavior by huddling and lying down"].

### F. Environmental Lighting (Camera Confound)
*   **Issue:** Overhead heat lamps in farrowing pens cast harsh directional shadows on piglet faces.
*   **Observation:** These shadows artificially mimic tension lines (e.g., cheek tension, snout bulge, lip contraction), tricking both human observers and computer vision models [@di-giminiani-2016; q="light source provided by a lamp suspended over the observation arena may have caused unwanted shadowing of the ventral aspect of the face"].

### G. Camera Angle & 3D Topography
*   **Issue:** Front-profile images are excellent for Orbital Tightening and Ear Position but terrible for Lip Contraction and Snout Angle. 
*   **3D Mapping Failure:** Attempts to use 3D Landmark Geometric Morphometrics failed to identify pain reliably due to the lack of sufficient surface texture/features on piglet faces to reconstruct 3D meshes of the ears and forehead [@lou-2022; q="lips and noses were moderately reconstructed, while the ears and forehead failed reconstruction for most models"].