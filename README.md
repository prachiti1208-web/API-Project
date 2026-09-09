\# Explainable Rubric-Grounded Handwritten Answer Evaluation



\##  Project Overview



This project presents an AI-based system for evaluating handwritten student answers using Computer Vision, Vision Transformers, Optical Character Recognition (OCR), and semantic Natural Language Processing (NLP).



The system is designed to detect answer regions from handwritten examination sheets, recognize handwritten text, compare student responses with reference answers, and generate an explainable evaluation.



The project focuses on making automated answer evaluation more transparent by providing not only a score, but also information about semantic similarity, matched concepts, missing concepts, and keyword overlap.



\---



\##  Objectives



\* Detect handwritten answer regions from student answer sheets.

\* Recognize handwritten text using OCR.

\* Process handwritten short answers using a Vision Transformer-based OCR model.

\* Evaluate MCQ answers using exact matching.

\* Evaluate descriptive answers using semantic similarity and rubric/concept matching.

\* Provide explainable evaluation results.

\* Expose the evaluation system through a FastAPI backend.



\---



\##  System Architecture



```text

Student Answer Sheet

&#x20;       │

&#x20;       ▼

PDF / Image Preprocessing

&#x20;       │

&#x20;       ▼

YOLO Answer Region Detection

&#x20;       │

&#x20;       ▼

Answer Region Cropping

&#x20;       │

&#x20;       ▼

TrOCR Handwriting Recognition

(Vision Transformer)

&#x20;       │

&#x20;       ▼

Structured Student Answers

&#x20;       │

&#x20;       ├───────────────┐

&#x20;       ▼               ▼

&#x20;  MCQ Evaluation   Short Answer Evaluation

&#x20;  Exact Matching   Semantic + Rubric Matching

&#x20;       │               │

&#x20;       └───────┬───────┘

&#x20;               ▼

&#x20;       Explainable Results

&#x20;               │

&#x20;               ▼

&#x20;         FastAPI / JSON

```



\---



\##  Project Workflow



\### 1. Dataset Preparation



Student examination PDFs are converted into page images and organized for processing.



\### 2. Annotation



Students 1–6 were manually annotated to identify answer regions.



\### 3. YOLO Training



A YOLO-based object detection model was trained using the annotated answer regions.



\### 4. Answer Region Detection



The trained YOLO detector is applied to student answer sheets to identify and crop answer regions.



\### 5. Handwriting Recognition



Cropped handwritten regions are processed using the pretrained:



`microsoft/trocr-base-handwritten`



TrOCR is based on a Vision Transformer architecture and is used as the handwriting recognition component.



\### 6. Answer Mapping



Recognized responses are organized according to question numbers and answer types.



\### 7. Answer Evaluation



\#### MCQ Questions



Questions 1–20 are evaluated using exact answer matching.



\#### Short Answers



Questions 21–35 are evaluated using:



\* Sentence semantic similarity

\* Rubric/concept matching

\* Keyword overlap

\* Combined scoring



\### 8. Explainability



For short-answer evaluation, the system can provide:



\* Semantic similarity score

\* Matched concepts

\* Missing concepts

\* Keyword overlap

\* Final evaluation decision

\* Explanation of the result



\### 9. API



The evaluation functionality is exposed through a FastAPI application.



Available endpoints include:



\* `GET /`

\* `GET /health`

\* `GET /results`

\* `GET /results/{student\_id}`

\* `POST /evaluate`



Interactive API documentation is available through FastAPI Swagger UI.



\---



\## Technologies Used



| Technology            | Purpose                      |

| --------------------- | ---------------------------- |

| Python                | Core development             |

| YOLO / Ultralytics    | Answer region detection      |

| TrOCR                 | Handwritten text recognition |

| Vision Transformer    | OCR architecture             |

| Sentence Transformers | Semantic similarity          |

| NLP                   | Short-answer evaluation      |

| FastAPI               | REST API                     |

| OpenCV                | Image processing             |

| PyTorch               | Deep learning                |

| Pandas                | Data processing              |

| NumPy                 | Numerical processing         |

| scikit-learn          | Machine learning utilities   |



\---



\##  Project Structure



```text

exam\_ocr\_project/

│

├── annotations/

│   ├── Student\_1\_annotations.json

│   ├── Student\_2\_annotations.json

│   ├── Student\_3\_annotations.json

│   ├── Student\_4\_annotations.json

│   ├── Student\_5\_annotations.json

│   └── Student\_6\_annotations.json

│

├── metadata/

│   ├── Question.txt

│   ├── Student\_MCQ.csv

│   ├── Teacher\_manual\_marks\_Anonymized.csv

│   ├── answerkey.txt

│   └── file.txt

│

├── scripts/

│   ├── annotate.py

│   ├── api.py

│   ├── check\_annotations.py

│   ├── convert\_pdfs.py

│   ├── detect\_7-50\_student.py

│   ├── detect\_and\_crop.py

│   ├── evaluate\_answers.py

│   ├── extract\_annotated\_crops.py

│   ├── final\_project\_demo.py

│   ├── map\_short\_answer.py

│   ├── ocr\_pipeline.py

│   ├── prepare\_yolo\_dataset.py

│   ├── recognize\_short\_answer.py

│   ├── test\_yolo\_data.py

│   ├── train\_yolo.py

│   └── visualize\_annotations.py

│

├── .gitignore

├── requirements.txt

└── student\_answers.json

```



\---



\##  Evaluation Method



The evaluation engine combines multiple signals for descriptive answers.



```text

Combined Score =

0.55 × Semantic Similarity

\+ 0.30 × Concept/Rubric Score

\+ 0.15 × Keyword Score

```



This allows the system to evaluate answers based on their meaning and important concepts rather than relying only on exact text matching.



\---



\##  Explainable Evaluation



Instead of returning only:



```text

Score: 1

```



the system can provide information such as:



```text

Semantic Similarity: 0.82

Concept Score: 0.75

Keyword Overlap: 0.60



Matched Concepts:

\- supervised learning

\- labeled data

\- known input/output



Missing Concepts:

\- none



Result:

Correct

```



This makes the evaluation process easier to interpret.



\---



\## 🚀 FastAPI



The API can be started using:



```bash

python -m uvicorn scripts.api:app --reload

```



The interactive Swagger documentation is available at:



```text

http://127.0.0.1:8000/docs

```



The API provides access to the project results and evaluation functionality through JSON responses.



\---



\## Current Limitations



The current prototype has limitations in handwritten OCR and automatic question-to-answer mapping.



The YOLO detection and TrOCR components have been integrated and tested, but handwriting recognition can produce noisy text for some answer regions.



Therefore, the current prototype should be considered a research/academic proof of concept rather than a production-ready automated examination system.



The downstream semantic and rubric-based evaluation has been validated using structured student-answer data.



\---



\##  Future Work



Future improvements include:



\* Fine-tuning the handwriting recognition model on domain-specific handwriting.

\* Improving question-number detection and answer mapping.

\* Improving YOLO region detection for different answer-sheet layouts.

\* Adding confidence scoring for OCR results.

\* Improving rubric generation and concept extraction.

\* Supporting more examination formats.

\* Adding a web-based teacher dashboard.

\* Adding database storage for evaluation results.

\* Providing downloadable evaluation reports.

\* Improving explainability with visual highlighting of relevant answer concepts.



\---



\##  Project Context



This project was developed as an academic/research project for handwritten answer evaluation using AI, Computer Vision, Vision Transformers, and NLP.



The project demonstrates how multiple AI components can be combined into an explainable automated evaluation pipeline.



\---



\## Disclaimer



This project is intended for academic and research purposes.



It should not be used as the sole basis for high-stakes academic grading without human verification.



