# Automated Scoring System for Assessing Swiss Municipalities Sustainability.


## Abstract
This thesis presents a research project to develop and implement an automated scoring sys-
tem to assess the Environmental, Social, Governance (ESG) factors across Swiss munici-
palities. Initially designed as a framework applicable to all municipalities in Switzerland,
we apply it to certain medium-sized cities: Nyon, Rolle, and Vevey. By fine-tuning large
language models based on the transformers architecture, we classified municipal council
transcripts into ESG categories and subsequently applied sentiment analysis to derive ESG
ratings. Our methodology included creating a new balanced dataset translated into French
from a collection of English article headlines and fine-tuning CamemBERT models. Multiple
models were produced using High-Performance Computing (HPC), and a weighted voting
scheme was employed to combine and enhance classification accuracy.

The results indicated strong performance in identifying environmental, governance, and
non-ESG-related content, with some challenges in distinguishing social aspects. Key find-
ings revealed a significant representation of environmental topics and a notable increase
in governance discussions. The social category may be underrepresented, possibly due to
overlaps with the non-ESG category. Emerging trends showed stable ratings for Nyon and
Vevey, while Rolle exhibited slightly more variability due to the lower number of available
council transcripts.

The automated system demonstrated efficiency in analysing large volumes of municipal
transcripts, providing valuable insights for policymakers. Limitations included occasional
misclassification of social content and a focus on French-language data. By translating the
dataset into German or Italian, the framework could be broadened to encompass the entire
country. This work offers a novel approach for automatic ESG assessment in the public
sector in Switzerland, facilitating more informed decision-making and setting a benchmark
for future research in automated ESG classification.

 



