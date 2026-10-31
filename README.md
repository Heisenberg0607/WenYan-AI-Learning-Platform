# An LLM Fine-Tuning and Semantic Search Platform for Classical Chinese Learning

[English](README.md) | [한국어](README.ko.md) | [简体中文](README.zh-CN.md)

A course project for *Text Mining Technologies* that integrates fine-tuned GPT-4o-mini models and BERT-based semantic search through FastAPI to support secondary school students learning Classical Chinese.

## Project Overview

### Problem

Secondary school students can struggle to understand Classical Chinese vocabulary and find verses they only partially remember. They also need timely feedback when practicing sentence segmentation and translation, so this project aimed to support these learning activities in a single platform.

### Approach

GPT-4o-mini models were separately fine-tuned for sentence segmentation and translation using 19,644 WenyanGPT samples and 5,013 WenyanBench Translation samples, respectively. The platform combines BERT-Ancient-Chinese semantic search over 910 classical poetry and prose entries with a vocabulary explanation scoring module using BERT-Base-Chinese and rule-based entity and relation extraction, integrated through FastAPI.

### What You Can Do

Users can add punctuation to unpunctuated Classical Chinese, translate it into modern Chinese, and search for works containing verses with similar meanings to their input. In vocabulary quizzes, users can explain a word in context and review comparisons with reference answers and automated scoring feedback.

## Key Results

| Component | Results and scope of evidence |
| --- | --- |
| Sentence segmentation | Split into 18,000 training and 1,644 validation samples. Train Accuracy of 0.968 (96.8%) at Step 2,998; Train Loss of approximately 0.099 stated in the report text |
| Classical Chinese translation | A dedicated model built using 5,013 parallel samples. The report describes late-training Accuracy at approximately 0.7 (about 70%), with fluctuations |
| Semantic search | Sliding-window and vector-similarity search implemented over 910 classical poetry and prose entries |
| Vocabulary quizzes | Random question selection, comparison with reference answers, and automated scoring based on entity semantic matching and relation errors |
| Feature integration | Sentence segmentation, translation, search, and vocabulary quizzes integrated through FastAPI |

The segmentation training accuracy and translation Accuracy are training-process metrics. The approximately 70% figure for translation is an approximate description in the report, rather than a human-assessed proportion of correct translations.

**My role:** I handled data preprocessing and JSONL construction, independently fine-tuned and validated the segmentation and translation models, and contributed to BERT semantic search and end-to-end platform integration.

## Feature Screenshots and User Flows

### 1. Classical Chinese Sentence Segmentation

Users enter unpunctuated text and view the result with punctuation added.

![Unpunctuated text input for sentence segmentation](assets/images/segmentation-input.png)

**Text input — Report Figure 1:** The screen shows the input “昔年无偶去今春又独归故人恩义重不忍更双飞”. Users request segmentation with the button below the input field.

![Sentence segmentation output](assets/images/segmentation-output.png)

**Output — Report Figure 2:** The result includes punctuation, as in “昔年无偶去，今春又独归。故人恩义重，不忍更双飞。”, along with a copy button.

### 2. Classical Chinese Translation

Users enter Classical Chinese text and view its modern Chinese translation alongside the original.

![Classical Chinese translation input](assets/images/translation-input.png)

**Translation input — Report Figure 4:** The screen accepts Classical Chinese text for translation. The same verse used in the segmentation example is entered with punctuation.

![Original Classical Chinese text and modern Chinese translation](assets/images/translation-output.png)

**Translation output — Report Figure 5:** The original appears on the left and the generated modern Chinese translation on the right. This output example allows users to compare the original wording with the model’s interpretation.

### 3. Semantic Search for Similar Verses

Searching a verse returns related passages together with the work’s title, historical period, author, and text.

![Initial web interface for semantic verse search](assets/images/search-home.png)

**Search interface — Report Figure 6:** Navigation at the top provides access to segmentation, translation, search, and vocabulary quizzes. The main area contains a query field and search button.

![Semantically similar verses and associated work information](assets/images/search-results.png)

**Search results — Report Figure 7:** The query “秋雁南飞无意住，关山角声四起时” returns works including Fan Zhongyan’s 《渔家傲·秋思》 and Meng Haoran’s 《秋登兰山寄张五》. Candidate passages appear together with information about each work.

### 4. Vocabulary Quizzes and Automated Scoring

The following two images show the question prompt and analysis output included in the report.

![Vocabulary quiz showing a verse and the word to explain](assets/images/annotation-question.png)

**Question prompt — Report Figure 8:** Users are asked to explain “燕草” in Li Bai’s 《春思》. They write an explanation based on the verse’s context.

![Entity and relation comparison between user and reference answers with scoring output](assets/images/annotation-feedback.png)

**Answer analysis — Report Figure 9:** The output displays the user and reference answers, extracted key concepts, semantic similarities between entities, and relation comparisons. In this example, the entity matching score is 94.5 points, with a 0-point relation error penalty, yielding 94.5/100 and a “correct” judgment. This is a scoring example for one answer.

## Personal Contributions

- **Training data preparation:** Preprocessed the segmentation and translation data and constructed JSONL files for fine-tuning.
- **Segmentation model:** Independently fine-tuned GPT-4o-mini and performed model validation.
- **Translation model:** Independently fine-tuned GPT-4o-mini and performed model validation.
- **Semantic search:** Contributed to the implementation of BERT-based semantic search.
- **Platform integration:** Contributed to end-to-end integration connecting the individual features.

## Data and Preprocessing

| Dataset | Size | Composition and use |
| --- | ---: | --- |
| WenyanGPT punctuation data | 19,644 samples | Split into 18,000 training and 1,644 validation samples after preprocessing |
| WenyanBench Translation | 5,013 samples | Parallel Classical Chinese originals and modern Chinese translations |
| Web-collected classical poetry and prose | 910 entries | Includes metadata such as title, author, historical period, text, translation, and annotations; shared by search and vocabulary quizzes |

The segmentation data was organized into instruction, input, and output fields, with multiple instruction formulations. It was then converted to messages-based JSONL with system, user, and assistant roles representing task instructions, original text, and reference output.

Translation data used the same conversational JSONL structure. The system role contained translation constraints, the user role the Classical Chinese original, and the assistant role the corresponding modern Chinese translation.

## Methodology and System Architecture

### 1. Task-Specific GPT-4o-mini Supervised Fine-Tuning

Segmentation was formulated as adding appropriate punctuation to the original text. The report explains that supervised fine-tuning was selected to adapt the model to the required output format and punctuation insertion rules.

A separate model was fine-tuned for translation using parallel Classical Chinese and modern Chinese data. Preserving meaning and information while maintaining consistent translation phrasing was the design goal. Both tasks used the OpenAI fine-tuning API.

### 2. BERT-Ancient-Chinese Semantic Search

A model pretrained on Classical Chinese was selected to match the language characteristics of the search corpus. Classical poetry and prose was divided into sliding windows, embeddings were formed through mean pooling, and cosine similarity was computed between query and candidate vectors.

The report describes difficulties locating passages for short queries when entire works were represented by single vectors, and exploration of half-line, one-line, and two-line segmentation. The final structure created overlapping spans using punctuation as boundaries and stored them in Window_1~Window_4 according to length.

The number of punctuation marks in a query determines which vector store is searched. Duplicate entries from the same work are removed from the highest-similarity candidates before returning Top-K results from distinct works.

### 3. Vocabulary Explanation Scoring

Questions are randomly selected from annotations associated with the 910 classical poetry and prose entries. Entities are constructed from user and reference answers using Jieba word segmentation and part-of-speech tagging, noun-focused candidate extraction, stopword and length filters, and rules.

Relation extraction uses 8 predefined patterns, including definition, attribute, and geographical location. BERT-Base-Chinese represents extracted entities as 768-dimensional vectors for semantic similarity calculation.

- **Entity matching:** For each reference entity, the most similar user entity is selected. A cosine similarity of at least 0.7 counts as a valid match.
- **Relation errors:** A relation is judged erroneous when its type matches the reference, its subject similarity is at least 0.7, and its object similarity is below 0.7.
- **Total score:** Mean similarity of valid matches × 100 − number of relation errors × 15 points. The minimum score is 0, and a score of at least 70 is judged “correct”.

### 4. Platform Integration

Sentence segmentation, translation, semantic search, and vocabulary quizzes were integrated through FastAPI. Learners can carry out individual learning activities by entering text, viewing results, writing answers, and reviewing scoring feedback.

## Evaluation and Results

### Segmentation Model

| Item | Reported information |
| --- | --- |
| Data split | 18,000 training / 1,644 validation samples |
| Observation point | Step 2,998 |
| Train Accuracy | 0.968 = 96.8% |
| Train Loss | Approximately 0.099 — stated in the report text |
| Interpretation scope | Metrics observed on training data |

![OpenAI fine-tuning training accuracy curve for the segmentation model](assets/images/segmentation-training-accuracy.png)

**Training curve — Report Figure 3:** The tooltip shows Step 2,998 and Train Accuracy of 0.968. This image displays the Accuracy curve; the approximately 0.099 Train Loss is stated in the report text.

### Translation Model

| Item | Reported information |
| --- | --- |
| Total data | 5,013 parallel Classical Chinese and modern Chinese samples |
| Late-training Accuracy | Approximately 0.7 (about 70%), with fluctuations |
| Evidence | Description of the training process in report Section 2.2.3 |
| Interpretation scope | A training-process reference metric that is difficult to interpret as an overall translation quality score |

Multiple translations can be appropriate for the same original, so Accuracy alone is insufficient to assess meaning preservation or naturalness. The report proposes evaluation combining BLEU, ROUGE, and human review as follow-up work.

Report Table 3 presents BLEU-1 0.4340, BLEU-2 0.2795, BLEU-3 0.1999, and BLEU-4 0.1481 for one sample as examples of data fields. These values belong to that example and do not represent average performance over the full evaluation data.

### Semantic Search and Vocabulary Explanation Scoring

The report’s query and returned-result examples demonstrate semantic search operation. Quantitative evaluation would require a query set with relevance judgments and measures such as Precision and Recall.

The 94.5/100 shown in the vocabulary scoring output is a score assigned to one user answer. The scoring system’s own accuracy and agreement with human assessments require separate evaluation.

## Limitations and Future Directions

- **Generalization evaluation:** The segmentation model’s 96.8% is training accuracy. Boundary-level metrics and error analysis on separate evaluation data are needed to assess performance on actual inputs.
- **Translation quality evaluation:** Criteria and human review for meaning preservation, omissions, and appropriateness of expression need to be supplemented.
- **Search coverage:** Selecting one window store based on query length can miss relevant passages in stores for other span lengths. Multi-window search and hybrid lexical–vector retrieval are future directions for consideration.
- **Search usability:** Initial model loading delays and limitations in handling modern Chinese queries were reported.
- **Scoring rules:** Entity and relation extraction depends on rules, which need refinement to handle unusual expressions, literary allusions, and complex paraphrases.
- **Personalized learning:** Combining difficulty information and learning history with the current random question selection is a future direction.
