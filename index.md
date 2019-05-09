
# Contents: Data Science Projects

## 1. [Classification Predictions of Uranium Enrichment With Real-World Experimental Data](CoincidenceModel/CoincidenceModel_Clean.md)
This notebook takes experimental gamma-ray measurements of uranium fission products I made during my Master's Thesis and creates classification models to discriminate between natural uranium, low-enriched uranium, and high-enriched uranium. These models are then used to determine which spectral features are most strongly correlated with uranium enrichment while being consistent across a one month range of decay times.

## 2. [Classification and Regression Predictions of Uranium Enrichment With Simulated Data](SCALE_05_2019/SCALE_4_30_2019.md)
This notebook builds classification and regression models for the determination of uranium and plutonium based on gamma-ray signatures across ranges of decay times. The simulated data of gamma-ray emissions is from irradiated uranium and plutonium (isotopes U-235, U-238, Pu-239, and Pu-240), from 180 decay times ranging from zero to eighteen days post-irradiation. This is important because conventional analytic techniques for enrichment determination rely on accurate decay time information. Furthermore, the developed models are capable of discerning new features within the data which are indicative of the actinide of interest, thus improving on the state-of-the-art methods for actinide determination in irradiated matrices.

## 3. [Front to Back Example Data Science Project](StateFarm/StateFarm.md)
This notebook explains and performs all of the major steps necessary to build robust machine learning models. A large dataset full of missing values, typos, etc. containing both numerical and categorical data is analyzed in a binary classification task. The steps necessary for data preparation including data cleansing, scaling, and imputing are performed and explained. Then, multiple classification models are fine-tuned via grid searches and the performance of different algorithms is compared. The final model obtains a score (measured via receiver operating characteristic area under curve) of over 0.98.

# About Me

I am a PhD candidate at the University of Texas at Austin in the Nuclear and Radiation Engineering Program. My dissertation research utilizes data science and machine learning prediction techniques to characterize special nuclear materials which are important for nuclear safeguards and nuclear nonproliferation. 

In my free time I like to mess around with the data science tools I've gathered to tackle new problems.

[Here](Adam_Drescher_CV.pdf) is my CV.

![Image](38011685_10215498402880927_7228843852281413632_o.jpg)





