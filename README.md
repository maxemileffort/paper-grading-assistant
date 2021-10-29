![cover_photo](./readme_files/cover_photo.jpg)
# Paper Grading Assistant

*Grading is the one of the top problems facing teachers in the US (Source: my teacher wife.) While it varies from subject to subject, many teachers spend upwards of several hours checking on the work of their students, to make sure they are growing and becoming prepared for their years ahead.*

*Because they are often times over-worked and under-paid, the goal of this project is to help teachers reclaim some of that time and money by making the paper grading process more efficient. The result of doing that leads to more meaningful feedback for the students and more productive use of time for the teachers.*

## 1. Data

For this, there are 2 sources of data. The first: [Hewlett Foundation Kaggle Competition](https://www.kaggle.com/c/asap-aes/data). Here, I used this data to sort of prove the concept as well as test the idea.

The other set of data is from my teacher wife, who provided some student papers in order to test the idea in more of a production-esque setting.

## 2. Data Cleaning

Placeholder

## 3. EDA

Data was explored more and more as the project went on, so EDA is in a lot of different places. The following are some of the interesting snippets pulled from the ongoing analysis, as well as where that can be found.

<!-- ![eda_conclusion](./readme_files/eda_conclusion.jpg) -->

## 4. Method

The dataset has different scales for each essay set, which were all standardized as a score out of 100. These were then all given a "Letter Grade" (A, B, C, D, or F), which is the class that we are trying to predict.

Additionally, Topic Modeling was used to examine the similarities between paragraphs. That was used to help a teacher leave feedback about how well a student organizes his thoughts.

More on these explained in the next section.

The idea here, is to sort a class of papers into piles quickly, so that teachers will know who needs more attention vs the ones that just need quick input. English teachers generally still want to read the papers, and this doesn't take away from that. Instead it helps teachers focus on the enjoyable parts, and makes the grading process more efficient and effective.

## 5. Algorithms & Machine Learning

Placeholder
<!-- I tested several different regression models from [SciKit Learn](https://scikit-learn.org/stable/)'s regression models. After finding a few that were close in performance, tuning brought about some bigger disparities that made Gradient Boost the winner for the filtering function of the algorithm and Ada Boost the winner of the player picking part of the algorithm.

>***NOTE:** I chose RMSE to control for dealing with outliers in the data. It's hard to consistently find players that score more than 30, 40 points, and some weeks they don't even occur. So RMSE would, because we are taking the root of those errors, control for those random high performers a little better.* -->

## 6. Pseudo-Coldstart Threshold

Placeholder
<!-- **Coldstart Threshold**: Recommender systems have a unique issue: *what does the algorithm recommend to new users when it has very little or no prior data?* 

Due to trades, player releases, the Draft, and some other stuff, there can be a lot of changes from season to season. So while we aren't using a recommender system, there's still the issue of picking players at the beginning of the season with the data that we *do* have.

I decided there were 2 options:

- Use last year's season data as a whole to make a decision for week 1 of this season.
- Wait for week 1's data in order to begin during week 2.

Because of how draft's work and the copious amounts of trades during the off-season, I decided to wait til week 1 was over and begin the process then. -->

## 7. Predictions

Placeholder

## 8. Future Improvements

* Placeholder

* Placeholder

## 9. Credits

Thanks to Kaggle for the free data, my wife for her guidance, and Mukesh Mithrakumar for being an extrememly knowledgeable and patient Springboard mentor.

Cover photo by [Green Chameleon](https://unsplash.com/@craftedbygc?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/study?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText). 