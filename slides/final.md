% Progress Report for Project Delta
% Victor Kong, Ce Li, Anna Liu, Weidong Qin, Yunfei Xia
% Dec 3, 2015

#BACKGROUND

## The Paper

- From OpenFMRI.org (ds005)
- "The Neural Basis of Loss Aversion in Decision-Making Under Risk"
  - by Sabrina M. Tom et al. (2007) in Science

## The Data

- 16 subjects, 1 task per subject, 3 runs per task, 1 behavioral data, 
  3 nerual condition files per subject
        - Condition 1: parametric gain
        - Condition 2: parametric loss
        - Condition 3: euclidean distance from gain/loss gamble matrix
- Examination of the neural systems that process decision utility with fMRI data
- Task:
  - Subjects offered 50/50 wager
  - Varying potential gains/losses
  - Prompted for decision to accept or decline



# Preprocessing: convolution-based smoothing with a Gaussian filter

\begin{figure}[!ht]
    \centering
    \includegraphics[width=120mm]{images/smooth_fig.png}
    \caption{Compare two bold images before and after smoothing}
    \label{fig:smoothing}
\end{figure}

# Convolution: Convolving predicted neural time course with hrf
\begin{figure}[!ht]
    \centering
    \includegraphics[width=120mm]{images/convolution3cond.png}
    \caption{Convolution}
    \label{fig:convolution results}
\end{figure}

# Linear Regression: multiple regression 

# PCA: detecting outliers and dimension reduction

# Hypothesis Testing: General t-test and locate activated brain region

\begin{figure}[!ht]
    \centering
    \includegraphics[width=120mm]{images/beta3condition_v1.png}
    \caption{Best estimates for 3 stimulus}
    \label{fig:Hypothesis testing results}
\end{figure}
\quad
\begin{figure}[!ht]
    \centering
    \includegraphics[width=120mm]{images/beta3condition_v2.png}
    \caption{Best estimates for 3 stimulus}
    \label{fig:Hypothesis testing results}
\end{figure}

# Logistic Regression: predict accept/reject decision with estimated betas


