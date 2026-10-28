---
annotations_creators:
- crowdsourced
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- hate-speech-detection
- sentiment-classification
- multi-label-classification
pretty_name: measuring-hate-speech
tags:
- arxiv:2009.10277
- counterspeech
- hate-speech
- text-regression
- irt
dataset_info:
  features:
  - name: comment_id
    dtype: int32
  - name: annotator_id
    dtype: int32
  - name: platform
    dtype: int8
  - name: sentiment
    dtype: float64
  - name: respect
    dtype: float64
  - name: insult
    dtype: float64
  - name: humiliate
    dtype: float64
  - name: status
    dtype: float64
  - name: dehumanize
    dtype: float64
  - name: violence
    dtype: float64
  - name: genocide
    dtype: float64
  - name: attack_defend
    dtype: float64
  - name: hatespeech
    dtype: float64
  - name: hate_speech_score
    dtype: float64
  - name: text
    dtype: string
  - name: infitms
    dtype: float64
  - name: outfitms
    dtype: float64
  - name: annotator_severity
    dtype: float64
  - name: std_err
    dtype: float64
  - name: annotator_infitms
    dtype: float64
  - name: annotator_outfitms
    dtype: float64
  - name: hypothesis
    dtype: float64
  - name: target_race_asian
    dtype: bool
  - name: target_race_black
    dtype: bool
  - name: target_race_latinx
    dtype: bool
  - name: target_race_middle_eastern
    dtype: bool
  - name: target_race_native_american
    dtype: bool
  - name: target_race_pacific_islander
    dtype: bool
  - name: target_race_white
    dtype: bool
  - name: target_race_other
    dtype: bool
  - name: target_race
    dtype: bool
  - name: target_religion_atheist
    dtype: bool
  - name: target_religion_buddhist
    dtype: bool
  - name: target_religion_christian
    dtype: bool
  - name: target_religion_hindu
    dtype: bool
  - name: target_religion_jewish
    dtype: bool
  - name: target_religion_mormon
    dtype: bool
  - name: target_religion_muslim
    dtype: bool
  - name: target_religion_other
    dtype: bool
  - name: target_religion
    dtype: bool
  - name: target_origin_immigrant
    dtype: bool
  - name: target_origin_migrant_worker
    dtype: bool
  - name: target_origin_specific_country
    dtype: bool
  - name: target_origin_undocumented
    dtype: bool
  - name: target_origin_other
    dtype: bool
  - name: target_origin
    dtype: bool
  - name: target_gender_men
    dtype: bool
  - name: target_gender_non_binary
    dtype: bool
  - name: target_gender_transgender_men
    dtype: bool
  - name: target_gender_transgender_unspecified
    dtype: bool
  - name: target_gender_transgender_women
    dtype: bool
  - name: target_gender_women
    dtype: bool
  - name: target_gender_other
    dtype: bool
  - name: target_gender
    dtype: bool
  - name: target_sexuality_bisexual
    dtype: bool
  - name: target_sexuality_gay
    dtype: bool
  - name: target_sexuality_lesbian
    dtype: bool
  - name: target_sexuality_straight
    dtype: bool
  - name: target_sexuality_other
    dtype: bool
  - name: target_sexuality
    dtype: bool
  - name: target_age_children
    dtype: bool
  - name: target_age_teenagers
    dtype: bool
  - name: target_age_young_adults
    dtype: bool
  - name: target_age_middle_aged
    dtype: bool
  - name: target_age_seniors
    dtype: bool
  - name: target_age_other
    dtype: bool
  - name: target_age
    dtype: bool
  - name: target_disability_physical
    dtype: bool
  - name: target_disability_cognitive
    dtype: bool
  - name: target_disability_neurological
    dtype: bool
  - name: target_disability_visually_impaired
    dtype: bool
  - name: target_disability_hearing_impaired
    dtype: bool
  - name: target_disability_unspecific
    dtype: bool
  - name: target_disability_other
    dtype: bool
  - name: target_disability
    dtype: bool
  - name: target_politics_alt_right
    dtype: bool
  - name: target_politics_communist
    dtype: bool
  - name: target_politics_conservative
    dtype: bool
  - name: target_politics_democrat
    dtype: bool
  - name: target_politics_green_party
    dtype: bool
  - name: target_politics_leftist
    dtype: bool
  - name: target_politics_liberal
    dtype: bool
  - name: target_politics_libertarian
    dtype: bool
  - name: target_politics_republican
    dtype: bool
  - name: target_politics_socialist
    dtype: bool
  - name: target_politics_other
    dtype: bool
  - name: target_politics
    dtype: bool
  - name: annotator_gender
    dtype: string
  - name: annotator_trans
    dtype: string
  - name: annotator_educ
    dtype: string
  - name: annotator_income
    dtype: string
  - name: annotator_ideology
    dtype: string
  - name: annotator_gender_men
    dtype: bool
  - name: annotator_gender_women
    dtype: bool
  - name: annotator_gender_non_binary
    dtype: bool
  - name: annotator_gender_prefer_not_to_say
    dtype: bool
  - name: annotator_gender_self_describe
    dtype: bool
  - name: annotator_transgender
    dtype: bool
  - name: annotator_cisgender
    dtype: bool
  - name: annotator_transgender_prefer_not_to_say
    dtype: bool
  - name: annotator_education_some_high_school
    dtype: bool
  - name: annotator_education_high_school_grad
    dtype: bool
  - name: annotator_education_some_college
    dtype: bool
  - name: annotator_education_college_grad_aa
    dtype: bool
  - name: annotator_education_college_grad_ba
    dtype: bool
  - name: annotator_education_professional_degree
    dtype: bool
  - name: annotator_education_masters
    dtype: bool
  - name: annotator_education_phd
    dtype: bool
  - name: annotator_income_<10k
    dtype: bool
  - name: annotator_income_10k-50k
    dtype: bool
  - name: annotator_income_50k-100k
    dtype: bool
  - name: annotator_income_100k-200k
    dtype: bool
  - name: annotator_income_>200k
    dtype: bool
  - name: annotator_ideology_extremeley_conservative
    dtype: bool
  - name: annotator_ideology_conservative
    dtype: bool
  - name: annotator_ideology_slightly_conservative
    dtype: bool
  - name: annotator_ideology_neutral
    dtype: bool
  - name: annotator_ideology_slightly_liberal
    dtype: bool
  - name: annotator_ideology_liberal
    dtype: bool
  - name: annotator_ideology_extremeley_liberal
    dtype: bool
  - name: annotator_ideology_no_opinion
    dtype: bool
  - name: annotator_race_asian
    dtype: bool
  - name: annotator_race_black
    dtype: bool
  - name: annotator_race_latinx
    dtype: bool
  - name: annotator_race_middle_eastern
    dtype: bool
  - name: annotator_race_native_american
    dtype: bool
  - name: annotator_race_pacific_islander
    dtype: bool
  - name: annotator_race_white
    dtype: bool
  - name: annotator_race_other
    dtype: bool
  - name: annotator_age
    dtype: float64
  - name: annotator_religion_atheist
    dtype: bool
  - name: annotator_religion_buddhist
    dtype: bool
  - name: annotator_religion_christian
    dtype: bool
  - name: annotator_religion_hindu
    dtype: bool
  - name: annotator_religion_jewish
    dtype: bool
  - name: annotator_religion_mormon
    dtype: bool
  - name: annotator_religion_muslim
    dtype: bool
  - name: annotator_religion_nothing
    dtype: bool
  - name: annotator_religion_other
    dtype: bool
  - name: annotator_sexuality_bisexual
    dtype: bool
  - name: annotator_sexuality_gay
    dtype: bool
  - name: annotator_sexuality_straight
    dtype: bool
  - name: annotator_sexuality_other
    dtype: bool
  splits:
  - name: train
    num_bytes: 53193054
    num_examples: 135556
  download_size: 20097387
  dataset_size: 53193054
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

## Dataset Description

- **Homepage:** http://hatespeech.berkeley.edu
- **Paper:** https://arxiv.org/abs/2009.10277

# Dataset card for _Measuring Hate Speech_

This is a public release of the dataset described in Kennedy et al. (2020) and Sachdeva et al. (2022), consisting of 39,565 comments annotated by 7,912 annotators, for 135,556 combined rows. The primary outcome variable is the "hate speech score" but the 10 constituent ordinal labels (sentiment, (dis)respect, insult, humiliation, inferior status, violence, dehumanization, genocide, attack/defense, hate speech benchmark) can also be treated as outcomes.  Includes 8 target identity groups (race/ethnicity, religion, national origin/citizenship, gender, sexual orientation, age, disability, political ideology) and 42 target identity subgroups, as well as 6 annotator demographics and 40 subgroups. The hate speech score incorporates an IRT adjustment by estimating variation in annotator interpretation of the labeling guidelines.

This dataset card is a work in progress and will be improved over time.

## Key dataset columns

 * hate_speech_score - continuous hate speech measure, where higher = more hateful and lower = less hateful. > 0.5 is approximately hate speech, < -1 is counter or supportive speech, and -1 to +0.5 is neutral or ambiguous.
 * text - lightly processed text of a social media post
 * comment\_id - unique ID for each comment
 * annotator\_id - unique ID for each annotator
 * sentiment - ordinal label that is combined into the continuous score
 * respect - ordinal label that is combined into the continuous score
 * insult - ordinal label that is combined into the continuous score
 * humiliate - ordinal label that is combined into the continuous score
 * status - ordinal label that is combined into the continuous score
 * dehumanize - ordinal label that is combined into the continuous score
 * violence - ordinal label that is combined into the continuous score
 * genocide - ordinal label that is combined into the continuous score
 * attack\_defend - ordinal label that is combined into the continuous score
 * hatespeech - ordinal label that is combined into the continuous score
 * annotator_severity - annotator's estimated survey interpretation bias

## Code to download

The dataset can be downloaded using the following python code:

```python
import datasets 
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()
df.describe()
```

## Citation

```
@article{kennedy2020constructing,
  title={Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application},
  author={Kennedy, Chris J and Bacon, Geoff and Sahn, Alexander and von Vacano, Claudia},
  journal={arXiv preprint arXiv:2009.10277},
  year={2020}
}
```

## Contributions

Dataset curated by [@ck37](https://github.com/ck37), [@pssachdeva](https://github.com/pssachdeva), et al.

## References

Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). [Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application](https://arxiv.org/abs/2009.10277). arXiv preprint arXiv:2009.10277.

Pratik Sachdeva, Renata Barreto, Geoff Bacon, Alexander Sahn, Claudia von Vacano, and Chris Kennedy. 2022. [The Measuring Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data Perspectivism](https://aclanthology.org/2022.nlperspectives-1.11/). In *Proceedings of the 1st Workshop on Perspectivist Approaches to NLP @LREC2022*, pages 83–94, Marseille, France. European Language Resources Association.