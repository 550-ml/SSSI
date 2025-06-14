# 📚 SciFigSeg Dataset

In the revised version, we have added a dedicated section titled “SciFigSeg Dataset”. Illustrations were automatically extracted from arXiv papers using pdffigures2, followed by manual filtering to retain complex illustrations, such as multi-stage frameworks, and system pipelines. We define a subregion as a visually and semantically independent and complete area, typically representing a functional module or a processing stage. Such regions often exhibit a consistent background color or a clearly defined outer boundary. Notably, arrows, connecting lines, and decorative elements are excluded from annotation. Ground truth annotations were created independently of our proposed method to avoid bias. We also clarify all illustration categories in the dataset. The “Other” category in Table 1 refers to hybrid diagram styles that do not strictly fit the flowchart or composite panel classification. We believe these revisions address the concerns regarding the transparency, and applicability of our dataset.

[download](https://aistudio.baidu.com/datasetdetail/337604)


## Clarification on heuristic components and threshold design.
In Sections 3.2, 3.3, and 3.4, we indeed employ a few hand-tuned thresholds to guide segmentation and alignment. The text box merging threshold $\tau_d$ (50 pixels) was chosen based on the output behavior of PaddleOCR, which returns one bounding box per line of text. Multiple lines often belong to the same semantic unit and appear in close proximity. A lower threshold may lead to fragmented text regions, while a higher threshold may incorrectly merge unrelated text. Through empirical testing, we found 50 pixels to be a good balance.
    The mask quality threshold $\tau$ (0.9) is determined by the confidence scores output by the SAM model. When the automatically generated prompt (e.g., point or box) is inaccurate, the resulting mask often has a lower score, reflecting poor quality. Thus, we discard masks with scores below 0.9. Conversely, masks with scores above 0.9 are typically well-structured and of high quality, and are retained as final outputs.

