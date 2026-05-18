# Sheep & Lamb Pain Recognition & Grimace Scale Manual

This document serves as a comprehensive cheatsheet and manual for the automatic and human-expert recognition of pain and holistic affective states in sheep (*Ovis aries*) and lambs. It synthesizes the State-of-the-Art (SoTA) pain assessment scales, Facial Action Units (FAUs), life-stage differences, common confounding factors, and broader visual signals of well-being.

---

## 1. State-of-the-Art (SoTA) Methods for Pain Assessment

Pain assessment in sheep relies heavily on specific, validated behavioral and facial scales. The literature identifies three primary facial grimace scales and one highly validated composite behavioral scale.

### 1.1 Facial Expression / Grimace Scales
1. **SPFES (Sheep Pain Facial Expression Scale)**: Developed for adult sheep (e.g., assessing naturally occurring pain like footrot and mastitis). Uses 5-6 FAUs. [@mclennan-2019; q="Development of a facial expression scale using footrot and mastitis as models of pain in sheep"]
2. **SGS (Sheep Grimace Scale)**: Developed for laboratory/adult sheep (e.g., post-operative distress after osteotomy). Uses 3 core FAUs, notably including the "Flehming" response as a unique pain indicator. [@hager-2017; q="The Sheep Grimace Scale as an indicator of post-operative distress"]
3. **LGS (Lamb Grimace Scale)**: Specifically developed for juvenile sheep (lambs) undergoing routine painful husbandry procedures (e.g., tail docking). Uses 5 FAUs. [@guesgen-2016; q="Coding and quantification of a facial expression for pain in lambs"]

### 1.2 Composite Behavioral Scale (Highest Evidence)
* **USAPS (Unesp-Botucatu sheep acute pain scale)**: A multidimensional behavioral scale considered the gold standard (high COSMIN evidence rating) for acute and post-operative pain in sheep. It assesses Interaction, Locomotion, Head Position, Posture, Activity, and Appetite. [@tomacheuski-2023; q="The UCAPS, UPAPS and USAPS showed the highest overall strength of evidence"]

---

## 2. Facial Action Units (FAUs) & Visual Pain Signals

Grimace scales generally score features on a 3-point scale: **0 = Not present**, **1 = Moderately present**, **2 = Obviously present**. 
*Note: The SGS incorporates a score of 3 specifically for Flehming.*

### 2.1 Orbital Tightening (Eyes)
* **Description**: Narrowing of the eye aperture, squinting, or complete closure of the eye.
* **Adult Sheep (SGS/SPFES)**: Half-closed eyes indicate moderate pain; completely closed eyes indicate severe pain. [@hager-2017; q="half-closed eyes were defined as an expression of “moderate” pain (score 1), whereas completely closed eyes were assigned with “severe” pain (score 2)"] Automated pain detection characterizes the ovine pain face as having "eyes partly closed". [@mota-rojas-2025; q="ears flipped and pinna not visible, with a “V” shaped nose, and eyes partly closed"]
* **Lambs (LGS)**: Lambs show "squeezing" of the eye. If eye closure reduces visibility by more than half, it is scored as obvious (2). [@guesgen-2016; q="If the eye closure reduces the visibility of the eye by more than half, it would be scored as obvious (2)"]

### 2.2 Ear Posture
* **Adult Sheep**: Ears rotate downwards, ventrally, and caudally (flattened). [@hager-2017; q="flattened ears and a slanted head were defined as an expression of moderate pain"] Automated systems look for "ears flipped and pinna not visible". [@mota-rojas-2025; q="ears flipped and pinna not visible, with a “V” shaped nose, and eyes partly closed"]
* **Lambs**: Ears are tense and point *backwards* or downwards so that the inner part of the ear is not visible. [@mclennan-2019; q="The ears of lambs point backwards when in pain, whereas in adult sheep the ears rotate ventrally and caudally"]

### 2.3 Cheek / Masseter Muscle Changes
* **Adult Sheep**: The masseter (cheek) muscle becomes prominent and tense. [@mclennan-2019; q="whereas in adult sheep the masseter muscle becomes more prominent"]
* **Lambs**: Cheeks exhibit *flattening*. Less bulging of the nose and cheek area; in obvious cases, the cheek has a "hollowed" appearance. [@mclennan-2019; q="cheek area in lambs being flattened, whereas in adult sheep the masseter muscle becomes more prominent"]

### 2.4 Nose and Mouth Features (Muzzle/Philtrum)
* **Adult Sheep (SPFES)**: Changes in lip and jaw profile, nostril, and philtrum shape.
* **Adult Sheep (SGS - Flehming)**: The Flehming response (curling back the upper lip exposing the teeth) is a sign of severe pain and is assigned a unique score of 3. Puckered lips express moderate pain (score 1). [@hager-2017; q="Flehming represents a sign of severe pain"]
* **Lambs (LGS - Nose)**: Nose appears tightened with a decrease in nostril size. May be depicted through 'flattening' (horizontal line) or 'pointing' ('V' shaped rather than 'U' shaped). [@guesgen-2016; q="pointing makes the nose appear more ‘V’ rather than ‘U’ shaped in frontal headshots"]
* **Lambs (LGS - Mouth)**: Lips are flattened and tightened into a horizontal line. Lack of the 'upwards curl' (smiling appearance). [@guesgen-2016; q="There is lack of the ‘upwards curl’ at the edge of the lips that gives lambs their ‘smiling’ appearance"]

### 2.5 Head Posture (USAPS / SGS)
* **Description**: Head carriage drops below the normal resting position.
* **Adult Sheep**: Hanging head is defined as severe pain (score 2). [@hager-2017; q="hanging ears and head were defined as severe (score 2)"] In the USAPS, head at the height of the withers is a score of 1, and head below the withers is a score of 2. [@silva-2025; q="Head below the withers. 2"]

---

## 3. Composite Body Behaviors & Holistic Affective States

Beyond nociceptive pain, monitoring requires identifying positive welfare, neutral resting, and non-pain negative affective states (fear/distress).

### 3.1 Negative_Nociceptive (Pain) - USAPS Scale
For automated systems, whole-body kinematics provide extremely high diagnostic accuracy (AUC ~0.91). [@silva-2025; q="The area under the curve of 0.91 demonstrates the high discriminatory capacity of the scale"]
1. **Posture**: Arched back, extended head and neck, lying down with head resting on the ground, moving the tail quickly and repeatedly (except when breastfeeding). [@silva-2025; q="The most frequent postures after surgery were ‘arched back’ and ‘moves tail’"]
2. **Interaction**: Apathetic, isolated, not interacting with other animals, not interested in the environment.
3. **Locomotion / Activity**: Moves about with restriction, short steps, pauses, lameness. Lying down and getting up frequently (restlessness), or reluctant to move/get up.
4. **Appetite**: Hyporexia or anorexia. However, literature notes that "appetite can be excluded without affecting the USAPS cut-off point" as it has low item-total correlation for acute orthopedic pain. [@silva-2025; q="The item appetite can be excluded without affecting the USAPS cut-off point"]

### 3.2 Negative_Affective (Stress/Fear/Distress)
* **Freezing**: A hallmark fear response where the "Animal remains motionless when disturbed. Body is stiff (non-relaxed), and animal highly alert with attention directed towards cause of disturbance." [@nc3rs-ethogram; q="Animal remains motionless when disturbed. Body is stiff (non-relaxed), and animal highly alert"]
* **Fleeing**: Moves rapidly away from potential or real threat.
* **Agonism**: Threat/thrust movements, pushing/boxing, or defensive crouching. [@nc3rs-ethogram; q="Threat/thrust (rapid head and fore-body movements towards another animal); upright or sideways posture"]

### 3.3 Positive_Welfare (Joy/Enrichment)
* **Play**: Highly energetic but non-aggressive social interaction. "Play behaviour contains the same behavioural elements as agonism, but is less intense, does not include injurious biting". [@nc3rs-ethogram; q="Play behaviour contains the same behavioural elements as agonism, but is less intense, does not include injurious biting"]
* **Allogrooming**: Affiliative behavior indicating social cohesion. "Gentle grooming (sometimes reciprocal) of any part of another individual. Engaging in allogrooming is voluntary". [@nc3rs-ethogram; q="Gentle grooming (sometimes reciprocal) of any part of another individual. Engaging in allogrooming is voluntary"]

### 3.4 Neutral_Resting (Baseline)
* **Stationary Alert**: The animal is "Alert (eyes open) but with no directed attention, while sitting, standing, or leaning against the cage side or an object within the cage. Body relaxed and immobile." [@nc3rs-ethogram; q="Alert (eyes open) but with no directed attention, while sitting, standing, or leaning... Body relaxed and immobile."]
* **Sleeping/Resting**: Normal lying with eyes closed, distinguished from pain by the lack of tension in facial muscles and relaxed body posture.

---

## 4. Edge-Cases, Confounders, and Trick Situations

### 4.1 Restraint and Camera Angle (Contextual)
* **Issue**: Holding or restraining a sheep alters its facial expression independently of pain (e.g., ears might go forward to gather information, overriding the pain signal).
* **Impact**: "restraint had some effect on facial expression and influenced both qualitative and quantitative measures of that expression." Furthermore, 2D angles affect measurements: "Observers may be able to intuitively correct for slight variations in camera angle or perceive depth within a photograph while the software cannot." [@guesgen-2016; q="restraint had some effect on facial expression and influenced both qualitative and quantitative measures"]

### 4.2 Sedation, Sleep, and Anesthesia (Physiological)
* **Issue**: Sedatives (e.g., ketamine, midazolam, isoflurane) or normal deep sleep states relax facial muscles or cause atypical postures, mimicking orbital tightening or altered ear positions.
* **Impact**: "False positives are known to occur in a small range of scenarios such as sedation/anaesthesia, sleeping status, or during bouts of aggression". [@cohen-2020; q="False positives are known to occur in a small range of scenarios such as sedation/anaesthesia, sleeping status, or during bouts of aggression"] Furthermore, "Behavior changes caused by sedation and pain may be similar, therefore sedation may be a confounding factor". [@silva-2025; q="Behavior changes caused by sedation and pain may be similar, therefore sedation may be a confounding factor"]

### 4.3 Coat Type / Hair / Breed (Morphological)
* **Issue**: Thick fleece, long hair, or dark coloration obscures FAUs like cheek flattening or orbital tightening.
* **Impact**: "observers had more difficulty assigning scores to long-haired ferrets... difficulties in assessing the facial expression of longer-haired, muscular, darker-haired animals". [@mclennan-2019; q="difficulties in assessing the facial expression of longer-haired, muscular, darker-haired animals"] Automated tracking requires diverse training sets to handle fleece occlusion.

### 4.4 Aggression and Agonistic Encounters (Contextual)
* **Issue**: During agonistic encounters, sheep may pull their ears back or squint to protect their eyes, directly mimicking a pain grimace.
* **Impact**: Aggressive interactions generate a False Positive for nociceptive pain due to identical facial motor responses deployed for protection rather than internal distress. [@cohen-2020; q="False positives are known to occur in a small range of scenarios such as sedation/anaesthesia, sleeping status, or during bouts of aggression"]

### 4.5 Coping Styles (Reactive vs. Proactive)
* **Issue**: Prey animals often conceal pain. Some individuals adopt a "reactive" coping style characterized by immobility and passivity rather than overt grimacing.
* **Impact**: Relying *only* on one observation point may fail to capture fluctuating pain. Interval/continuous monitoring is required. [@guesgen-2016; q="displaying a reactive coping style to pain, which is characterized by immobility and passivity"]