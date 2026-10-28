# Datasets

## SemEval 2023 Task 3: Propaganda Detection ✅
https://propaganda.math.unipd.it/semeval2023task3/

Data description
- News genre categorization: given a news article, determine whether it is an opinion piece, aims at objective news reporting, or is a satire piece. This is a multi-class (single-label) task at article-level. 
- Framing detection: given a news article, identify the frames used in the article. This is a multi-label task at article-level. A frame is the perspective under which an issue or a piece of news is presented. We consider 14 frames: Economic, Capacity and resources, Morality, Fairness and equality, Legality, constitutionality and jurisprudence, Policy prescription and evaluation, Crime and punishment, Security and defense, Health and safety, Quality of life, Cultural identity, Public opinion, Political, External regulation and reputation. This taxonomy, as well as a discussion on the definitions of frame, For details on the definition of frame and the taxonomy used in our annotations, we followed (Card et al., 2015). Specifically, 
- Persuasion Techniques Detection: Given a news article, identify the persuasion techniques in each paragraph. This is a multi-label task at paragraph level. Justification, simplification, distraction, call, manipulative wording, attack on reputation.
- All multiclass classification

Comments:
> Comparable to propaganda part of no_prop_overlap dataset but only multiclass though could be made multiple. Good topic fit but different length (article/paragraph instead of sentence). High fit.

## Measuring Hate Speech Dataset ✅
https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

Data description
- Hate speech score
- Ordinal labels sentiment, (dis)respect, insult, humiliation, inferior status, violence, dehumanization, genocide, attack/defense, hate speech benchmark
- Includes 8 target identity groups (race/ethnicity, religion, national origin/citizenship, gender, sexual orientation, age, disability, political ideology) and - 42 target identity subgroups, as well as 6 annotator demographics and 40 subgroups.
- hate_speech_score - continuous hate speech measure, where higher = more hateful and lower = less hateful. > 0.5 is approximately hate speech, < -1 is counter or supportive speech, and -1 to +0.5 is neutral or ambiguous.
- text - lightly processed text of a social media post
- comment_id - unique ID for each comment
- annotator_id - unique ID for each annotator
- sentiment - ordinal label that is combined into the continuous score
- respect - ordinal label that is combined into the continuous score
- insult - ordinal label that is combined into the continuous score
- humiliate - ordinal label that is combined into the continuous score
- status - ordinal label that is combined into the continuous score
- dehumanize - ordinal label that is combined into the continuous score
- violence - ordinal label that is combined into the continuous score
- genocide - ordinal label that is combined into the continuous score
- attack_defend - ordinal label that is combined into the continuous score
- hatespeech - ordinal label that is combined into the continuous score
- annotator_severity - annotator's estimated survey interpretation bias

Papers that use this data:
1. Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application. arXiv preprint arXiv:2009.10277. https://hatespeech.berkeley.edu/publication/01_main_paper/
    - "Scaling procedure integrates naturally with a multitask deep learning architecture" -> **They predict the label for each component using a multitask framework.**
    - Faceted Rasch Scaling: item response theory for annotated data.
    - Data from Youtube, Reddit, Twitter; stratified on how hateful the comment is likely to be.
    - They say their continuous measurement scale is capable of accommodating annotator perspective.
    - https://github.com/dlab-projects/hate_measure
        - The dataset is used largely as-is, with tokenization for training their transformer model.
        - hate_measure is a single-task regressor. It runs ModernBERT, averages the token embeddings, passes them through a 128-unit ReLU layer with 0.4 dropout, and outputs one number trained with MSE against hate_speech_score.
        - In the model, the log-odds of an annotator choosing rating k over k−1 is θₙ − δᵢ − αⱼ − τₖ. The terms are the comment’s hate score (θₙ), the item’s difficulty (δᵢ), the annotator’s severity (αⱼ), and the step difficulty (τₖ). Severity is subtracted, so a higher αⱼ lowers the chance of a more hateful rating.

2. Pratik Sachdeva, Renata Barreto, Geoff Bacon, Alexander Sahn, Claudia von Vacano, and Chris Kennedy. 2022. The Measuring Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data Perspectivism. In Proceedings of the 1st Workshop on Perspectivist Approaches to NLP @LREC2022, pages 83–94, Marseille, France. European Language Resources Association. https://aclanthology.org/2022.nlperspectives-1.11/
    - This paper introduced the dataset
    - https://github.com/dlab-projects/hate_measure_data/tree/main 
        - Data just used as is.
    - Faceted Rasch measurement (Linacre, 1994) allows one to capture multiple features (“facets”) that influence the generation of a label, including content of the data sample, the annotator’s perspective, and the task at hand.
    - Found low annotator agreement on the responses to each survey item using Kripendorff's alpha (1 - observed disagreement / expected disagreement). Annotators agree more on whether text is hate speech but less on what the specific component is.
    - Annotators could optionally identify their own demographic information: over 80% White, 56% women, 85% straight, nearly half had grad degrees.
    - Provides means to capture annotator perspective via severity parameters, so annotator agreement is not necessary.

Comments:
> Good mix of datatypes, both techniques of speech and entity recognition, similar to IAC _bi scores and IBM WA/WA_bi parts of no_prop_overlap. High fit.

## Comparative Agendas Project: Datasets & Codebooks ✅
https://www.comparativeagendas.net/datasets_codebooks

Data description
- Organizes government activities—such as laws, speeches, hearings, and debates—into a shared, standardized system of 21 major topics and over 220 subtopics.
- Political text coded against major topics and subtopics. CMP records position, CAP codes topic. Major/sub structure could use hierarchical task_structure. 

Comments:
> High fit with CMP.