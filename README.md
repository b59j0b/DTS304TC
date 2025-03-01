java cModule code and Title 
DTS304TC Machine Learning School Title 
School of AI and Advanced Computing Assignment Title 
Assessment Task 1 
DTS304TC Machine Learning 
Coursework – Assessment Task 1 
Submission deadline: TBD
Percentage in final mark: 50%
Learning outcomes assessed: 
A.   Demonstrate   a   solid   understanding   of   the   theoretical   issues   related   to   problems   that   machine   learning   algorithms   try   to   address.
B.   Demonstrate   understanding   of   properties   of   existing   ML   algorithms   and   new   ones.
C.   Apply   ML   algorithms   for   specific problems.
Individual/Group: Individual
Length: The   assessment   has   a   total   of   4   questions   which   gives   100   marks.   The   submitted   file   must   be   in   pdf   format.
Late policy: 5%   of   the   total   marks   available   for the   assessment   shall be   deducted   from   the   assessment mark for each working day after the submission date, up to a maximum of five working days 
Risks: 
•      Please   read   the   coursework   instructions   and   requirements   carefully. Not   following   these   instructions   and   requirements   may result   in   loss   of   marks.
•      The   formal   procedure   for   submitting   coursework   at   XJTLU   is   strictly   followed.   Submission   link   on   Learning   Mall   will   be provided   in   due   course.   The   submission   timestamp   on   Learning   Mall   will be      used   to   check   late   submission.
Question 1: Coding Exercise - Disease Classification with Machine Learning (80 Marks) In    this    coding   assessment,   you      are   presented   with   the      challenge   of    analyzing   a   dataset   that      contains   patient   demographics   and   health   indicators   to   predict   disease   classifications.   This   entails   solving   a   multi-   class   classification   problem   incorporating   both   categorical   and   numerical   attributes.Your    initial    task    is    to   demonstrate   proficiency   in   encoding   categorical   features   and   imputing   missing   values   to   prepare   the   dataset   for   training   a   basic   classifier.   Beyond   these   foundational   techniques,   you   are   invited   to showcase your advanced skills. This may include hyperparameter    tuning    using    sophisticated   algorithms   like   the   PBT   or   Bayesian   Optimization.   You   are   also   encouraged   to   implement   strategies   for   outlier   detection   and      handling,   model   ensembling,   and   addressing   class   imbalance   to   enhance      your   model's performance.Moreover, an   external      test set without ground   truth labels has   been   provided.   Your      classifier's   performance   will   be    evaluated   based   on   this    set,   underscoring   the   importance    of   building a   model   with   strong   generalization   capabilities.The      competencies you develop during this practical project are not only essential for successfully completing   this   assessment but   are   also   highly   valuable   for   your   future   pursuits   in   the   field   of data   science.   Throughout   this   project,   you   are   encouraged   to   utilize   code   that   was   covered   during   our   Lab   sessions,   as      well   as   other   online   resources   for   assistance.   Please   ensure   that   you   provide   proper   citations   and   links   to      any   external   resources   you   employ   in   your   work.
Important: You may   not use   ChatGPT   to   directly   generate   answers   for the   coursework.   It   is   important to   think   critically   and   develop   your   own   solutions—by yourself or   through reputable   online resources—to   strengthen   your research   and   critical   thinking   skills.   You may, however, use   ChatGPT for code understanding or debugging, as well as for grammar and writing assistance. If you do so, use it responsibly, ensuring you fully understand and take ownership of all the content in your coursework. 
(A) XGBoost and Hyperparameter Tweaking (28 marks) 
(A1) Basic Feature Preprocessing 
. Dataset Loading and Inspection: Load the   provided   training   dataset   from   the   CSV   file   into   a   panda   DataFrame   (or   a   similar   structure).
. Handling Categorical Features: Identify   any   categorical   features.   Encode these   features appropriately   (e.g.,   one-hot   encoding   or   label   encoding).   Justify   your   choice   of   encoding   method.
. Handling Missing Values: Identify   and   handle   missing   values   in   the   dataset.   Provide   a   concise   explanation   of   your   chosen   strategy   (e.g.,   imputation, removal   of   rows,   etc.).   You   can use   a simple   imputation   method   (mean/mode   imputation)   at   this   point.
. Dataset Splitting: Randomly   split the   data   into training,   validation   and   internal   test   subsets   for model   tuning. Note   that this   internal test   set   must   be   kept   separated   for internal   evaluation.
Important: The   same preprocessing   steps   (encoding, missing-value   handling,   etc.)   must be   consistently   applied to   all   data   subsets,   including the   external   test   set provided   later.
(A2) XGBoost Tweaking and Training 
1.         Hyperparameter Tuning Approach
o    Utilize   advanced hyperparameter   tuning   strategy   such   as   Bayesian   Optimization,   Population-Based   Training   (PBT),   or   a   similarly   sophisticated   method.
o      Clearly   describe   the   search   space   (i.e.,   which   hyperparameters   you   chose   to   tune   and   the   range   of   values   for   each).
o    Ensure   hyperparameter   tuning   is   performed   using   the   training   and   validation   sets—not   on   your   internal   test   set.
2.         XGBoost Model Training
o    Use   the   chosen   hyperparameter-tuning   technique   to   fit   the   XGBoost   model.
o      Log   or   record   key   information   about   the   tuning   process   (e.g., best   parameters   found,   final   objective   scores).
(A3) Evaluation and Reflection 
1.          Performance   on   Internal Test   Set. After   finalizing   your   hyperparameters,   retrain   the   model   on
the   combined   (training   + validation)   set.   Then   evaluate   on   the   internal   test   set.   Report   at   least   two metrics:   accuracy and macro-averaged F1 score.
2.          Explanation   of Tuning Principles. Briefly   explain   how your   chosen   advanced tuning   approach
works   (e.g.,   how   Bayesian   optimization   narrows   search   space,   or how   PBT   explores   hyperparameter   configurations,   etc.).
3.       Demonstrating   the   Importance   of   Hyperparameter   Tuning
o    Provide   a   simple   comparison   of   performance   results   between:   A   default   XGBoost   model
(no   advanced   tuning   or very   simple   tuning)   and   your   advanced   tuning   approach
o      Discuss   any   observed   improvements   or unexpected   findings.
(B) Additional Tweaking (20 marks) 
(B1) Additional Tweaking Implementation 
You   are   encouraged to   apply   at   least   two   extra   strategies to boost performance.   Example   strategies   may   include   (but   are   not   limited   to):
.               Using   alternative preprocessing   methods   (e.g.,   advanced missing value   imputation   strategy,   outlier   treatment,   advanced   feature   engineering)
.               Exploring   class imbalance   handling   (e.g.,   SMOTE,   class-weight   adjustments)
.               Exploring   alternative   classifiers   (e.g.,   random   forest   or   other   classifiers)
.               Building   ensembles   of   models   (e.g.,   combining XGBoost with   other   classifiers)
.                Any other innovative   approach relevant to   your   dataset
Please   clearly   document   each   additional   strategy   you   implement   and   show   enough   code/comments   so we   can   understand how   you   incorporated   it   into   your pipeline.
(B2) Additional Tweaking Evaluation and Reflection 
1.         Motivation and Principles
o    For   each   strategy,   explain   why   you   believed   it   might   improve   performance   (e.g.,   addressing   outliers,   class   imbalance,   or   feature   interaction代 写DTS304TC Machine Learning Assessment Task 1Java
代做程序编程语言s).
o    Elaborate   on   the   theoretical   or   conceptual   principle behind   your   chosen   extra   strategies   (e.g.,   how   does   SMOTE   handle rare   classes,   why   does   certain   feature   engineering help,   etc.).
2.         Results Reporting   and Analysis
o      Provide   results   (accuracy,   F1,   or   other   metrics)   indicating   whether   the   strategy   helped,   did   not help,   or had neutral   impact.   Present   all relevant   performance   metrics   in   a concise   table.
o      Offer possible   explanations   for   why   certain tweaks worked   or   did   not   work.
o      Explain   your   other   efforts   (whether   they   are   successful   or   not)   for   improving   the   classification performance.
(C) External Benchmarking and Final Result Reporting (12 marks) 
(C1) Final Result Reporting 
1.         Per-Class Precision, Recall,   Specificity, and F1. On   your   internal   test   set,   compute   and
display:   Precision,   Recall,   F1   score   for   each   class. A   confusion matrix to   visualize performance.
2.         Feature Importance Analysis.
o      For   the   same   model,   extract   feature-importance   scores   if supported   by your   chosen   classifier   (e.g., XGBoost).
o    Identify   the   top   three   most important   features   for   your   prediction   task.
o      If   your best   model   does not   natively   support   feature   importance,   you   may use   a   surrogate   technique   (e.g.,   SHAP,   LIME)   or   a   secondary model   for   demonstration.
(C2) External Benchmarking 
1.         Retraining on the Full Dataset
o    Retrain your   best-performing   classifier   on   the   entire   dataset   (i.e.,   training   +   validation   +   internal   test   sets).
o    Apply your   final preprocessing and   hyperparameter   configurations   consistently.
2.         Predicting on the External   Test   Set
o    Predict   on   dts304tc_a1_disease_dataset_external_test.csv,   which   does   not   include   ground   truth   labels.
o      Output probabilistic   scores   or predicted   labels   in   a   CSV named   external_test_results_[your_student_id].csv,   with:
.                First   column:   “Patient_ID”
.                Second   column:   Your   predicted   class   labels   (integers)
3.          Ranking   and Feedback. Your   submission   will   be   evaluated   against   an   external   “ground   truth” for   benchmarking.
(D) Challenges and Reflections (10 marks) 
.             Write   a   brief   reflection   highlighting   unique   or   interesting   algorithms   or   strategies   you   implemented   in   the   coursework.
.             Reflect   on   the   challenges   you   personally   faced   during   the   coursework,   describing   the   efforts   you   made   to   overcome   them   and   the   key   lessons   you   learned.
.             Based   on   your   reflections,   discuss   potential   future   work   that   could   further   enhance   semantic   segmentation   in   street-view   applications.
(E) Coding Quality, Answer Sheet Quality, and Submission Guidelines (10 marks) 
.               Submit   your   completed   worksheet   in   PDF   format.
.               Submit   your   Jupyter Notebook   in   .ipynb   format.   Your notebook must be   well-organized   and   include   clear   commentary   and   clean   code   practices.   All   code   execution   results   must   be   clearly   visible   and    match    the    results    provided    in   the   worksheet.    Additionally,   the   notebook      should   be      fully   reproducible—running it from start to   finish should not   produce   any   errors.
If you   have   written   supplementary   code   that   is   not   contained   within   the   Jupyter   notebook,   you   must   submit   that   as   well   to   guarantee   that   your   Jupyter   notebook   functions   correctly.
.             Submit   the   results   of   your   external   test   as   a   file   named   external_test_[your_student_id].csv.   This   CSV   file   must   be   correctly   formatted:   the   first   column   must   contain   patient   ID                            s,                and                   the                second   column   must   list   your   predicted   classification   labels.   Any   deviation   from   this   format   may   result   in   the   file   being   processable   by   our   grading   software,   and   therefore   unable   to   be   scored.
Project Material Access Instructions 
To   obtain   the   complete   set   of   materials   for   our   project,   including   the   dataset,   code,   and   Jupyter   notebook   template   files,   please   use   the   links   provided below:
(OneDrive   Link): https://1drv.ms/f/s!AoRfWDkanfAYoo4P7hiPebYwCnSlag?e=Jcg0WD
When prompted,   use   the   following password to   unlock the   zip   file:   DTS304TC   (please   note   that   it   is   case-   sensitive   and   should be   entered   in   all   capital   letters).
Please   note   that   the   primary   library   dependencies   for this   project   include   pandas,   scikit-learn,   xgboost,   and   the   ray   library   with   the   tune   module   enabled   (ray[tune]).
Question 2:       Analytical Questions (20 marks) 
Students   are   required   not   to   use   AI   models,   such   as   ChatGPT,   for   assistance   with   this   question.   You should give clear calculation steps and   explain the   relevant   concepts   using   your   own words.
(a)                      AdaBoost Algorithm (10 Marks) 
Consider   the   process   of   creating   an   ensemble   of   decision   stumps,   referred   to   as   Gm   ,   through   the   standard   AdaBoost   method.


The   diagram   above   shows   several   two-dimensional   labeled points   along   with   the   initial   decision   stump   we've   chosen.   This   stump   gives   out binary   values   and makes   its   decisions   based   on   a   single   variable   (the   cut-off).   In   the   diagram,   there's   a   tiny   arrow perpendicular to   the   classifier's   boundary   that   shows   where         the   classifier predicts   a   +1.   Initially,   every point   has   the   same   weight.
1.          Identify   all   the points   in   the   above   diagram   that   will   have   their   weights   increased   after   adding   the
initial   decision   stump   (adjustments   to   AdaBoost   sample   weights   after   the   initial   stump   is   used)   (2   marks)
2.          On   the   same   diagram,   draw   another   decision   stump   that   could   be   considered   in   the next   round   of
boosting.   Include   the   boundary   where   it   makes   its   decision   and   indicate   which   side   will   result   in   a   +1   prediction.   (2 marks)3.          Will   the   second   basic   classifier   likely   get   a   larger   importance   score   in   the   group   compared   to   the   first
one?   To put   it   another way, will   α2      >      α1   ?      Just   a   short   explanation   is   needed   (Calculations   are   not   required).   (3 marks)4.          Suppose   you   have   trained   two   models   on   the   same   set   of data:   one   with   AdaBoost   and   another   with   a
Random   Forest   approach.   The   AdaBoost   model   does   better   on   the   training   data   than   the   Random
Forest   model.   However,   when   tested   on   new,   unseen   data,   the   Random   Forest   model   performs   better.   What   could   explain   this   difference   in performance?   What   can   be   done   to make   the   AdaBoost model         perform. better?   (3   marks)
(b) K-Means and GMM Clustering (10 marks) 
1.          Reflect   on   the   provided   data   for training   and   analyze the   outcomes   of K-Means   and   GMM
techniques.   Can   we   anticipate   identical   centroids   from   these   clustering   methods   if   the   number   of   clusters   is   2   or   3?   Please   state   your   reasoning.   (4   marks)

2.          Determine   which   of   the   given   cluster   assignments   could be   a   result   of applying   K-means   clustering,
and   which   could   originate   from   GMM   clustering,   providing   detailed   explanations   for   your reasoning.   (6   marks)  








         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
