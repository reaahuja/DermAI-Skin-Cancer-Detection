\documentclass{article} % For LaTeX2e
\usepackage{iclr2022_conference,times}
% Optional math commands from https://github.com/goodfeli/dlbook_notation.
\input{math_commands.tex}

%######## APS360: Uncomment your submission name
%\newcommand{\apsname}{Project Proposal}
%\newcommand{\apsname}{Progress Report}
\newcommand{\apsname}{Final Report}

%######## APS360: Put your Group Number here
\newcommand{\gpnumber}{28}

\usepackage{hyperref}
\usepackage{url}
\usepackage{graphicx}
% additional packages
\usepackage{multirow}
\usepackage{array}
\usepackage{hhline} 
\usepackage{float}
\usepackage{adjustbox}
\hypersetup{hidelinks} % can toggle as needed
\hypersetup{
  colorlinks   = true, %Colours links instead of ugly boxes
  urlcolor     = black, %Colour for external hyperlinks
  linkcolor    = blue, %Colour of internal links
  citecolor   = blue %Colour of citations
}

%######## APS360: Put your project Title here
% Can change :) 
\title{External lesion classification for the early detection of skin cancers}

%######## APS360: Put your names, student IDs and Emails here
\author{Rea Ahuja  \\
Student\# 1009152373 \\
\texttt{rea.ahuja@mail.utoronto.ca} \\
\And
Christina Pizzonia \\
Student\# 1007914250 \\
\texttt{christina.pizzonia@mail.utoronto.ca} \\
\AND
Mai Shimozato  \\
Student\# 1009080959 \\
\texttt{mai.shimozato@mail.utoronto.ca} \\
\And
Halle Teh \\
Student\# 1008965997 \\
\texttt{halle.teh@mail.utoronto.ca} \\
\AND
}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to \LaTeX{} to determine where to break
% the lines. Using \AND forces a linebreak at that point. So, if \LaTeX{}
% puts 3 of 4 authors names on the first line, and the last on the second
% line, try using \AND instead of \And before the third author name.

\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}

\iclrfinalcopy 
%######## APS360: Document starts here
\begin{document}


\maketitle

\begin{abstract}
This project focuses on the implementation of a convolutional neural network with transfer learning to accurately diagnose external skin lesions from a single image. Source code for this project can be found \href{https://github.com/reaahuja/APS360_Project/tree/main}{\color{blue}{here}}.

%######## APS360: Do not change the next line. This shows your Main body page count.
----Total Pages: \pageref{last_page}
\end{abstract}

\section{Introduction}
Skin cancers are one of the most commonly diagnosed forms of cancer in Canada, with over 80,000 new diagnoses each year \citep{CancerCAN}. While the prognosis for skin cancers are positive if caught early (about a 99 \% 5-year survival rate), this number drops significantly (under 30\%) once the cancer has metastasized \citep{acj}. 

\subsection{Motivation}
This steep drop in survival rates makes early detection critical for improving the chances of successful treatment \citep{acj}. Dermatologists typically rely on visual inspection to initially screen for skin cancers. Unfortunately, this method is time-consuming, dependent on appointment availability and prone to misdiagnoses, especially in those with darker skin tones \citep{bias3}.

\subsection{Goals and importance}
To address these challenges, the proposed model will leverage computer vision technology to distinguish between benign conditions (nevi and seborrheic/lichen-planus-like keratoses) and potentially malignant tumors (melanomas and basal cell carcinomas) using images of external lesions (Fig.~\ref{fig:Intro_Model}). 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{Figs/into.png}
    \caption{The proposed model will return a label identifying the skin condition in the input image}
    \label{fig:Intro_Model}
\end{figure}

The goal is to provide an accurate (low false negative rate for cancers) and accessible method for classifying lesions on all skin types in real time, facilitating early intervention and improving patient outcomes. Utilizing this system in clinical settings has the potential to reduce the burden on dermatologists, acting as a screening tool to prioritize patients at higher risk \citep{screen}. 

\subsection{The need for a deep learning approach}
Convolutional neural networks (CNNs) are ideal for image classification tasks involving large, diverse datasets \citep{CNN1}. Given the variability in skin lesion appearance across different sexes, skin types and ages, the ability of neural networks to extract features and identify key patterns across comprehensive datasets is crucial. Current deep learning approaches may also be able to classify lesions more quickly and accurately than licensed professionals \citep{stan}. 

\section{Background \& related work}
Numerous systems for skin condition classification have been developed to address challenges associated with model accuracy. A subset of these systems are described below, providing valuable insights that inform the proposed approach. 

\subsection{Skin Lesion Classification using a K-NN}
This paper aimed to classify melanoma, nevus and seborrheic keratosis using a dataset of dermoscopic images \citep{K-NN}. An active contour model was employed to eliminate noise and isolate the relevant lesion area. After segmentation, features such as texture, shape and colour were extracted for analysis while K-Nearest Neighbours algorithm was used for the final classification. The model struggled with melanoma and seborrheic keratosis, primarily due to class imbalance and challenges in separating lesions from the background due  obstructions such as hair. As such, the proposed data processing steps will include functions for hair removal and data augmentation techniques to ensure a balanced training set is used. 

\subsection{Lesion Classification Using ResNet + CBAM Hybrid Approach}
This study employed a hybrid approach for image classification, using flipping and rotating for data augmentation, integrating ResNet50 for feature extraction and a Convolutional Block Attention Module (CBAM) with capsule network for improved classification accuracy relative to CNNs \citep{Hybrid}. This justifies the use of  CBAM in the model and the flipping/rotation of images applied during data augmentation.

\subsection{Binary Classification for Skin Cancer}
The authors proposed a skin lesion classification system based on kernel sparse representation and dictionary learning. The system classifies the input lesion as either melanoma or normal using colour and/or texture features. The methodology effectively reduced sensitivity to noise and varying image conditions (including light reflection, hair, air bubbles and frames)  \citep{bin}. The focus on reducing noise sensitivity for improved results underscores the importance of filtering images during pre-processing.

\subsection{Melanoma Detection using CNNs}
A 2022 study utilized a custom CNN for melanoma detection. The method included convolution layers, pooling layers, and fully connected layers for classification. The final model’s performance was tested against a Support Vector Machine (SVM), showing that the CNN outperformed the SVM in both validation accuracy and precision \citep{CNNex}. This study highlights the benefits of pooling and fully-connected layers in a classification network, reinforcing the selection of a CNN-based approach for this project and justifying the use of an SVM as the baseline model.

\subsection{U-Net and CNN Fusion Model}
This study proposed a fusion model that integrated a U-Net model and a CNN model for melanoma detection. The U-Net model was employed to segment and localize the skin lesions in the dermoscopy images. Following segmentation, the CNN model classified the segmented images into multiple categories. The fusion model, utilizing Adam and Adadelta optimizers, demonstrated superior performance across various parameters compared to many state-of-the-art techniques \citep{U-NET}. This study suggests that CNNs perform better when combined with segmented lesions, which informed the centering and cropping approach during pre-processing as well as trials with the Adam optimizer during model training. 

\section{Data processing}
A variety of dermoscopic images were selected and cleaned for training the proposed model.
\subsection{Sources of data}
All images were collected from two main sources, each with 7 classes: the “Human Against Machine with 10000 training images” (HAM10000) database with 10,015 images collected from both the Rosendahl clinic (Australia) and the ViDIR group at the Medical University of Vienna \citep{ham}, and the International Skin Imaging Collaboration (ISIC) 2019 Challenge training dataset, with 25,331 images from across Europe, Australia and North America \citep{ISIC}. 

The  model will select between the 4 largest, most common classes: mel (melanoma), nv (nevus), bcc (basal cell carcinoma) and bkl (benign keratosis-like lesions) (Fig.~\ref{fig:dist}).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.4\textwidth]{Figs/dist.png}
    \caption{Class distribution across the pooled HAM10000/ISIC datasets for model training}
    \label{fig:dist}
\end{figure}

A third dataset, sourced from the Hospital Italiano de Buenos Aires, will be used as additional testing data and will be discussed further in section \ref{sect8}.  

\subsection{Pre-processing pipeline}
The pre-processing pipeline  (Fig.~\ref{fig:labelled})  involves several steps to ensure data quality, quantity and uniformity before model training:

\begin{enumerate}
    \item Removal of duplicate images: The\texttt{   imagededup} Python package was used to detect and remove exact and duplicate images within the combined datasets \citep{r4}
    \item Removal of corrupted images: \texttt{PIL} (Python Imaging Library) scripts were used to detect and remove corrupted images
    \item Hair and marking removal: The “dull razor” technique, implemented by converting images to grayscale, applying a black hat filter to detect hair edges and using inpainting (\texttt{INPAINT-TELEA} in OpenCV) to fill regions masked by these edges \citep{dullrazor}
    \item Noise removal: Using OpenCV, Gaussian Blur, Median Filter, and Bilateral Filter were applied to smooth images and preserve important details while reducing noise artifacts
     \item Resizing and normalization: All images were resized to a consistent resolution of 224x224 pixels (for ResNet input) and normalized using \texttt{torchvision.transforms}
\end{enumerate}

\begin{figure}[H]
    \centering
    \includegraphics[width=1.0\textwidth]{Figs/labelled.png}
    \caption{Data processing pipeline}
    \label{fig:labelled}
\end{figure}


After processing, a 60:20:20 split was performed to create training, validation and testing sets. Due to the class imbalance in the original dataset, images from minority classes were randomly flipped and rotated (90 or 270 degrees) until all classes had roughly the same number of images (Fig.~\ref{fig:bal}) . 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{Figs/balanced.png}
    \caption{Class distribution before and after data augmentation was applied}
    \label{fig:bal}
\end{figure}


%To determine if methods should be employed to augment the minority classes or simply reduce the amount of majority class images for the final project, the proposed model was tested on both a full dataset and reduced dataset with an equal number of images per class (see \ref{challenge!}).



\section{Model architecture}
The current model uses 18-layer ResNet (with approximately 11.7 million parameters) for feature extraction, followed by a CNN (with approximately approximately 712,000 parameters) for classification. The CNN receives inputs from ResNet that are  \texttt{128 * 512 * 7 * 7}. The three convolutional layers have output channel widths of 128, 64 and 32 each, with a kernel size of 3 and padding size of 1. Each convolutional layer is directly followed by CBAM, batch normalization and dropout (50\%). The final prediction is made by a single linear, fully connected layer (\texttt{(32 * 7 * 7, 4)}) (Fig.~\ref{fig:model}).
 
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{Figs/model.png}
    \caption{A high-level overview of the proposed model architecture integrating ResNet-18 with CBAM and a custom CNN}
    \label{fig:model}
\end{figure}

In detail, CBAM consists of channel and spatial attention sub-modules that enable the network to focus on informative components:
\begin{enumerate}
    \item Channel Attention Module (CAM):
    CAM takes feature maps as input and applies global average and global max pooling operations generating feature descriptions. These descriptions are then passed through a shared MLP (multilayer perceptron) to produce channel attention maps. Lastly, the attention maps are element-wise multiplied with the input feature maps to enhance the important channels \citep{cb}.
    \item Spatial Attention Module (SAP): SAP uses the output of CAM as input, applies average pooling and max pooling operations along the channel axis. A convolutional layer with a sigmoid activation function is then applied to generate spatial attention maps, which was then element-wise multiplied with the input feature maps (similar to CAM) to focus on the important spatial locations \citep{cb}.
\end{enumerate}

The pre-trained, ResNet-18 model was modified to output feature maps instead of classification scores by removing its final fully connected layer and pooling layers. These feature maps, extracted from training and validation datasets, were used to train the custom CNN for final classification. 
%Note that the selection of ResNet over AlexNet (another popular image classification deep learning model) was deliberate: ResNet's residual learning framework performed about 4-6\% better, on average, when tested against AlexNet on the same datasets with the same CNN classifier.  

The three 2D convolutional layers, each followed by CBAM and batch normalization, were used in the CNN to enhance feature representation and stabilize training.

\section{Baseline model}
The proposed baseline model uses simple artificial neural network (ANN) for feature extraction and a support vector machine (SVM) for classification. 

The simple ANN architecture was designed with 3 fully connected layers (\texttt{(224 * 224 * 3, 512), (512, 128) and (128, 4)}) and utilized to extract high-level features from each image (Fig.~\ref{fig:base}).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{Figs/base.png}
    \caption{A high-level overview of the baseline model architecture using an ANN-SVM combination for image classification}
    \label{fig:base}
\end{figure}

Instead of passing the extracted features through a final fully connected layer with softmax activation for
classification, images were instead fed into an SVM classifier. This SVM uses the extracted features
from the simple ANN network to classify the skin conditions.

SVM was selected as the baseline due to its widespread use and proven performance in the classification of skin lesions \citep{b1} and its demonstrated excellence in classification tasks \citep{b2}. They are also advantageous in terms of simplicity and training speed, especially with smaller datasets.
They find the best possible hyperplane that separates different classes of data points. Despite these
advantages, SVMs may not capture the complex patterns and relationships as effectively as CNNs,
which learn features automatically during the training process. 

\section{Quantitative results}
During hyper-parameter turning, the team determined that the best results were achieved when the primary model was trained with a batch size of 64 and a learning rate of 0.001, introducing image rotations after each epoch to prevent memorization of the training set (Fig. ~\ref{fig:prim_results}). Loss was determined using Cross Entropy (CE) Loss and parameters optimized using Stochastic Gradient Descent (SGD) with momentum (0.9) and weight decay (1e-4). The final model reached training accuracy of 75.3\% and validation accuracy of 71.0\%, before over-fitting.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.4\textwidth]{Figs/acc.png}
    \caption{Training and validation accuracy curves from model training}
    \label{fig:prim_results}
\end{figure}

When evaluated on the 20\% test split from the HAM10000 and ISIC pooled datasets, the model achieved a similar accuracy of 73.7\%. This is a significant improvement from the baseline model, which reached an accuracy of 44.1\% on the same dataset. 

The primary model also achieved higher precision, recall and F1 scores  (all above 0.7) when compared to the baseline model (Fig.~\ref{fig:recall}). 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{Figs/r.png}
    \caption{Precision, recall and F1 scores of the baseline and primary models on the test split}
    \label{fig:recall}
\end{figure}

The importance of high recall (sensitivity) and precision is especially important since this model aims to classify potentially cancerous lesions. The model exhibits a higher recall than precision for all classes except mel, suggesting that the model is prone to making false positive predictions. This is the ideal case for a skin lesion classification model, as it means false negatives are minimized (i.e. few cancerous lesions go undetected). The low recall (0.7) and high precision (0.81) for mel will be addressed in the following section. 

\section{Qualitative results} 

To evaluate model performance, the team assessed several sample predictions made by the model (Fig.\ref{fig:samplePredictions}). 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{Figs/pred.png}
    \caption{Sample predictions from the primary model}
    \label{fig:samplePredictions}
\end{figure}


The model was able to correctly distinguish between  both the cancerous (mel, bcc) and non-cancerous (bkl, nv) lesion types, although initial images look similar to the human eye. However, the model was often liable to incorrectly classify cancers (mel, bcc) relative to the other classes, as in Fig.\ref{fig:samplePredictions}. This occurred despite applying data augmentation to bcc and mel, which were less common relative to nv in the original dataset.  

In addition, although quantitative results illustrate that the recall for mel in particular is lower than the precision, the sample predictions above suggest that when incorrectly classified, mel is most commonly assigned to the bcc class. That is, the model still outputs a condition that is cancerous, increasing the probability that the danger of the lesion would not go undetected. This is also evident in confusion matrix (Fig.~\ref{fig:confmatrix}) from the model on the test data: 68 mel samples would have ben labelled as bcc, over 24 and 60 for bkl and nv respectively. 


\begin{figure}[H]
    \centering
    \includegraphics[width=0.35\textwidth]{Figs/conf.png}
    \caption{Confusion matrix from the testing split from the HAM10000/ISIC pooled data}
    \label{fig:confmatrix}
\end{figure}

The model was also no more likely to classify a difficult image as "nv" over any other class, suggesting data augmentation was successful. Overall, the proposed model successfully identified true positives for the majority of input images, but further refinements are necessary to address its limitations in recognizing less represented classes. 


\section{Model evaluation on new data}
\label{sect8}
A new dataset of 1032 dermoscopic images from 623 patients from the Hospital Italiano de Buenos Aires, unseen during both training and validation phases, was selected to evaluate model performance \citep{newData}. The dataset contains diverse ethnicities from the datasets initially used to build the model (images are from individuals across South America, distinct from the HAM10000 and ISIC datasets which are predominantly European and Australian) \citep{messi}, minimizing any overlap or skin tone bias relative to the original dataset to ensure impartiality.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.35\textwidth]{output.png}
    \caption{Confusion matrix from the dataset from the HIBA dataset}
    \label{fig:conf}
\end{figure}

The performance metrics derived by the confusion matrix (Fig.~\ref{fig:conf}) on the HIBA dataset and additional statistics indicate that the model performs generally well on unseen data:

\begin{itemize}
    \item An accuracy of 68.9\% indicates that most of the model’s predictions are correct. This accuracy is similar to the accuracy (73.7\%) reported by the model on the testing split in the "Quantative Results" section
    \item An average precision and recall of 72\% and 69\% respectively  indicate that the model effectively balances its ability to classify true positives and minimize false positives. This balanced performance ensures the model is not only fairly accurate, but also reliable.
    \item An F1 score of 70\% further reflects the model’s consistent performance across these metrics, as an F1 score is the harmonic mean of precision and recall.
\end{itemize}

Note that confusion matrix highlights some difficulties with correctly identifying bcc relative to the other classes (which may also negatively skew the recall and precision values from above).  


\section{Discussion}
Model performance during inference highlights both the strengths of the current model and potential areas for improvement.

\subsection{Contextualizing model performance} 
The model achieved a testing accuracy of 68.9\% on unseen data, relative to a 73.7\% accuracy on the test split (20\%  of the original HAM10000/ISIC pooled data). Although these values may appear low, they are indicative of the difficulty of classifying lesion samples from images alone. Studies have shown that professional dermatologists are only able to accurately identify skin lesions about 53-55\% of the time \citep{stan}: this is far below the performance of the proposed model, and underscores the importance of deep learning models for lesion identification. 

\subsection{Under-performing classes}
Despite the use of data augmentation, the model struggled with bcc in particular, which had the fewest amount of samples in the original dataset (see Fig. \ref{fig:labelled}). This is not completely unsurprising, and suggests that although augmentation was performed via flipping and rotations, it may not completely address a lack of diversity in skin lesion appearance. This is especially relevant when samples share a high degree of visual similarity to begin with \citep{K-NN}. Rotated or flipped images of skin lesions can often resemble the originals, limiting the effectiveness of this technique in enhancing dataset diversity. 

That said, data augmentation was able to reduce bias towards the majority class, nevus (nv). When the model was uncertain about a cancerous lesion, it avoided defaulting to the majority class, instead distributing predictions across other classes. This was not the case for the baseline model (SVM), which typically skewed toward the majority class.

Future work should focus on collecting a broader range of diverse images, especially for cancerous lesions, rather than relying heavily on rotation/flipping to better capture lesion variability.

\subsection{Demographic Variability} 
The model's performance on the HIBA dataset was about 5\% lower than its performance on the HAM10000/ISIC images. This may be in part due to skin tone bias in the training and testing sets: datasets such as HAM10000 and ISIC predominantly represent European and Australian populations, while the HIBA dataset originates from South America \citep{data3}. The under-representation of darker skin tones in the training dataset makes it difficult for the model to generalize across different demographic groups \citep{bias2}. This demographic variation, combined with differences in image quality, likely contributed to the lower accuracy on the HIBA dataset, emphasizing the need for population diversity in medical image datasets.

Addressing challenges such as class imbalance, overfitting, and demographic variability is crucial for improving the model's robustness and generalizability across diverse patient populations.


\section{Ethical considerations}
Various ethical concerns may arise from inaccurate inferences or biases inherent in the original data.

\subsection{Lack of diversity in initial datasets}
One significant issue is skin tone bias in the initial datasets. Across the ISIC and HAM10000 datasets, very few images contain class 5 and 6 (darker) skin tones on the Fitzpatrick scale \citep{bias4}. Studies suggest that models trained predominantly with images from lighter skin tones are likely to perform worse when tested on darker skin tones \citep{bias2}, leading to disparities in detection accuracy across racial groups \citep{fair}. This bias is particularly concerning given the varied outcomes of skin cancer treatment across races, with black patients experiencing higher mortality rates when diagnosed with melanomas and carcinomas due to the late detection of malignancies \citep{race}. 

Another area of concern is the unequal distribution of classes within the datasets. The HAM10000 and ISIC 2019 \citep{data1} sets both under-represent rarer conditions, which may result in a final model biased towards detecting common conditions during inference \citep{overRep}. This limitation can lead to serious health implications, as malignant lesions, especially those with atypical presentations, may be more likely to be classified incorrectly.

\subsection{Dangers of model usage}
Since the end-goal is a skin lesion classification system for early cancer-screening, any incorrect predictions, especially false negatives, can have lasting impacts on users. A false negative during inference can significantly harm the patient in the long run, as skin cancers are often treatable but only when detected early \citep{falseNeg}. 

All input images to the model, including those used during inference, originate from real individuals and are classed as highly sensitive data. Mishandling or unauthorized access to these images could lead to significant privacy violations and the undermine patient trust. Therefore, it is crucial to ensure that all unnecessary metadata is removed and no individual can be identified from a single image \citep{danger}.

 
%%%%%%%%%%%%% SAMPLE FIGURES %%%%%%%%%%%%$
% \begin{figure}[h]
% \begin{center}
% \includegraphics[width=0.6\textwidth]{Figs/td-deep-learning.jpg}
% \end{center}
% \caption{Sample figure caption. Image: ZDNet}
% \end{figure}

% \subsection{Tables}

% All tables must be centered, neat, clean and legible. Do not use hand-drawn
% tables. The table number and title always appear before the table. See
% Table~\ref{sample-table}.

% Place one line space before the table title, one line space after the table
% title, and one line space after the table. The table title must be lower case
% (except for first word and proper nouns); tables are numbered consecutively.

%%%%%%% SAMPLE TABLE %%%%%%%%%%%
% \begin{table}[t]
% \caption{Sample table title}
% \label{sample-table}
% \begin{center}
% \begin{tabular}{ll}
% \multicolumn{1}{c}{\bf PART}  &\multicolumn{1}{c}{\bf DESCRIPTION}
% \\ \hline \\
% Dendrite         &Input terminal \\
% Axon             &Output terminal \\
% Soma             &Cell body (contains cell nucleus) \\
% \end{tabular}
% \end{center}
% \end{table}

\label{last_page}

\bibliography{APS360_ref}
\bibliographystyle{iclr2022_conference}

\end{document}
