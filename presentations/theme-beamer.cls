% Template for LUT presentations with LaTeX beamer.
%
% by Aleksei Mashlakov, mashlakov@gmail.com
%
% Inspired by the template for SINTEF presentations by Federico Zenith,
% which was derived (through several iterations) from Håvard Berland's
% beamerthementnu class.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

\ProvidesClass{theme-beamer}[2021/12/17]

\RequirePackage{etoolbox}
\newcommand{\@DashColor}{bluePolimi}
\newcommand{\@DashHeight}{1.5mm}

\newcommand{\testcolor}[1]{\colorbox{#1}{\textcolor{#1}{test}} \texttt{#1}}
\newcommand{\hrefcol}[2]{\textcolor{orange}{\href{#1}{#2}}}

\newbool{LUTdark}
\PassOptionsToClass{aspectratio=169}{beamer} % Default, may be overridden
\DeclareOption{cyandash}{\renewcommand{\@DashColor}{bluePolimi}}
\DeclareOption{nodash}{\renewcommand{\@DashHeight}{0mm}}
\DeclareOption{light}{\boolfalse{LUTdark}}
\DeclareOption{dark}{\booltrue{LUTdark}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass{beamer}

\usepackage{caladea,carlito}
\renewcommand{\familydefault}{\sfdefault}

\RequirePackage{graphicx,theme-color,ifthen}

% Following commands will not be activated for article mode.
\mode<presentation>

% No navigation symbols
\setbeamertemplate{navigation symbols}{}

% phantom items 
\setbeamercovered{transparent} % invisible

% nomenclature
\usepackage[symbols,nomain,acronyms,makeindex,section=subsection]{glossaries-extra}
\newglossary{latin}{lat}{latout}{Latin alphabet}
\newglossary{greek}{grk}{grkout}{Greek alphabet}
\newglossary{sub}{sub}{subout}{Subscripts}
\newglossary{sup}{sup}{supout}{Superscripts}
\newglossary{dimless}{diml}{dimlout}{Dimensionless numbers}
\newglossary{notation}{not}{notout}{Notation}
\glsnoexpandfields
\glstocfalse
% \makeglossaries
% \input{nomenclature.tex}

% packages
\usepackage{amsfonts,amsmath,oldgerm}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\minimize}{minimize} % minimize operator
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{subfig}
\usepackage{tikz}
\usepackage{verbatim}
\usepackage[utf8]{inputenc}
\usepackage{csquotes}

% line spacing
\linespread{1.1}\selectfont % <---

% item spacing
% \let\tempone\itemize
% \let\temptwo\enditemize
% \renewenvironment{itemize}{\tempone\addtolength{\itemsep}{0.35\baselineskip}}{\temptwo}

% bibliography 
\usepackage[style=numeric,backend=bibtex]{biblatex}
% \usepackage[autocite=footnote,style=authortitle,backend=bibtex]{biblatex} 
\addbibresource{references.bib}

% Set colours
\setbeamercolor{block title}{bg=bluePolimi, fg=white}
\setbeamercolor{block body}{bg=bluePolimi!10!white, fg=black}

\setbeamercolor{block title alerted}{bg=redPolimi, fg=white}
\setbeamercolor{block body alerted}{bg=redPolimi!10!white, fg=black}

\setbeamercolor{block title example}{bg=greenPolimi, fg=white}
\setbeamercolor{block body example}{bg=greenPolimi!10!white, fg=black}

\setbeamercolor{alerted text}{fg=bluePolimi}
\setbeamercolor{headline}{fg=white,bg=orange}
\setbeamercolor{author}{fg=black}
\setbeamercolor{example text}{fg=orange,bg=grey}
% \setbeamercolor{block title}{fg=white,bg=bluePolimi}
% \setbeamercolor{block body}{fg=black,bg=grey}
% \setbeamercolor{block body example}{fg=black,bg=grey}
\setbeamercolor{caption name}{fg=black}
\newcommand{\@LOGO}{logo_centrato_COL_positivo.png}
\ifbool{LUTdark}{
    \setbeamercolor{normal text}{fg=white}
    \setbeamercolor{background canvas}{bg=black}
    \setbeamercolor{structure}{fg=white}
    \setbeamercolor{title}{fg=white,bg=black}
    \setbeamercolor{footline}{fg=white} 
    \setbeamercolor{itemize items}{fg=bluePolimi}
    \renewcommand{\@LOGO}{logo_centrato_COL_negativo.png}
}{
    \setbeamercolor{normal text}{fg=black}
    \setbeamercolor{background canvas}{bg=white}
    \setbeamercolor{frametitle}{fg=black,bg=white}
    \setbeamercolor{structure}{fg=bluePolimi}
    \setbeamercolor{footline}{fg=black}
    \setbeamercolor{itemize items}{fg=bluePolimi}
    \renewcommand{\@LOGO}{logo_centrato_COL_positivo.png}
}

% font sizes
\usefonttheme[onlymath]{serif}
\setbeamerfont{author}{size=\footnotesize}
\setbeamerfont{date}{size=\scriptsize}
\setbeamerfont{caption}{size=\scriptsize}
\setbeamerfont{normal text}{size=\footnotesize}
\setbeamerfont{equation}{size=\footnotesize}
\setbeamerfont{subsection in toc}{size=\footnotesize}
\setbeamerfont{section in toc}{size=\normalsize}
\renewcommand*{\bibfont}{\footnotesize}
\setbeamertemplate{itemize/enumerate body begin}{\footnotesize}
\setbeamertemplate{itemize/enumerate subbody begin}{\scriptsize}

% Code to get prettier boxes
\setbeamertemplate{blocks}[rounded][shadow=true]

% Bullets in several levels
% \setbeamertemplate{itemize item}{\textbf{\normalsize\guillemotright}}
\setbeamertemplate{items}[triangle]
\setbeamertemplate{itemize subitem}{\normalsize\textendash}
\setbeamertemplate{itemize subsubitem}{\normalsize\textbullet}

% pink squares and triangle for items
\newcommand{\redsquareitem}{\item[\textcolor{pink}{\tiny$\blacksquare$}]}
\newcommand{\redtriangleitem}{\item[\textcolor{pink}{$\blacktriangleright$}]}

% to have a separate numbering for back-up slides 
\usepackage{appendixnumberbeamer}

% table of content in itemize style
\setbeamertemplate{section in toc}[square]
\setbeamertemplate{subsection in toc}[default]

% Define frame title and subtitle layout
\setbeamertemplate{frametitle}
{   \hbox{%
    \begin{beamercolorbox}[wd=.75\paperwidth,ht=17mm,dp=0mm, left]{frametitle}
        \usebeamerfont{frametitle}\MakeUppercase{\insertframetitle}\\
        \vspace{0.1ex}{\textcolor{\@DashColor}{\rule[0ex]{1cm}{\@DashHeight}} 
        \usebeamerfont{framesubtitle}\insertframesubtitle~\vphantom{phantomText}}\\
    \end{beamercolorbox}% 

    % ! REMOVED POLIMI LOGO FROM TOP RIGHT CORNER
    % \begin{beamercolorbox}[wd=.25\paperwidth,ht=15mm,dp=2mm, left]{frametitle}
    % \hfill{\vspace{1.5ex}\includegraphics[height=4ex]{\@LOGO} \hspace*{2em}}
    % \end{beamercolorbox}
    }%
    \vskip0pt%
}

% 
\newcommand{\setdepartment}[1]{\def\department{#1}}
\titlegraphic{\includegraphics[height=8ex]{\@LOGO}}

% Define the title page
\setbeamertemplate{title page}
{
  \hbox{%
   \begin{beamercolorbox}[wd=0.74\paperwidth,sep=8pt,left]{institute}
      \usebeamerfont{institute}{\department\par \insertinstitute}
    \end{beamercolorbox}
    \begin{beamercolorbox}[wd=0.2\paperwidth,sep=8pt,left]{institute}
     \usebeamercolor[fg]{titlegraphic}\hfill\inserttitlegraphic
    \end{beamercolorbox}
  }%
  \begin{centering}
    \begin{beamercolorbox}[sep=8pt,left]{title}
      \usebeamerfont{title}\MakeUppercase{\inserttitle}\par%
      \textcolor{\@DashColor}{\rule[0ex]{1cm}{\@DashHeight}} 
    \end{beamercolorbox}%
    \begin{beamercolorbox}[sep=8pt,left]{subtitle}
      \ifx\insertsubtitle\@empty%
      \else%
        \vskip0em%
        {\usebeamerfont{subtitle}\usebeamercolor[fg]{subtitle}\MakeUppercase{\insertsubtitle}\par}%
      \fi%     
    \end{beamercolorbox}
    \vskip0em\par
    \begin{beamercolorbox}[sep=8pt,left]{author}
        % \usebeamerfont{author}{Presented by \insertauthor}
        \usebeamerfont{author}{\insertauthor}
    \end{beamercolorbox}
    \begin{beamercolorbox}[sep=8pt,left]{date}
      \usebeamerfont{date}{\insertdate}
    \end{beamercolorbox}%\vskip0.5em
    \end{centering}
}


% Define the bottom of each standard frame
\setbeamertemplate{footline}{
    % bottom bar
    \begin{beamercolorbox}[wd=\paperwidth, ht=10pt, dp=1pt]{footline}
    \usebeamerfont{footline}
    
    % ! REMOVED DATE
    % \begin{tikzpicture}[remember picture,overlay]
    % \node[anchor=south west, 
    %       xshift=10mm, 
    %       yshift=.4mm] 
    %      at (current page.south west) 
    %      {\insertdate}; 
    % \end{tikzpicture}
    
    % ! REMOVED AUTHOR... CLEAN SLIDES, ADD IF NEEDED
    % \begin{tikzpicture}[remember picture,overlay]
    % \node[anchor=south west, 
    %       xshift=35mm, 
    %       yshift=.4mm] 
    %       at (current page.south west) 
    %      {\insertauthor,~\department,~\insertshortinstitute}; 
    % \end{tikzpicture}
    
    % ! REMOVED SHORT TITLE
    % \begin{tikzpicture}[remember picture,overlay]
    % \node[anchor=south east, 
    %       xshift=-15mm, 
    %       yshift=0.4mm] 
    %       at (current page.south east) 
    %      {\insertshorttitle}; 
    % \end{tikzpicture}

    \begin{tikzpicture}[remember picture,overlay]
    \node[anchor=south east, 
          xshift=-5mm, 
          yshift=0.5mm] 
         at (current page.south east) 
         {\insertframenumber \,/\, \inserttotalframenumber};
    \end{tikzpicture}
    
    \end{beamercolorbox}
}

% ! REMOVED NAVIGATION SYMBOLS
% add navigation symbols
% \setbeamercolor{mycolor}{fg=white}%,bg=structure!30}
% \addtobeamertemplate{footline}{
%     \leavevmode%
%     \hbox{%
%     \begin{beamercolorbox}[wd=\paperwidth,ht=2.75ex,dp=.5ex,right,rightskip=5mm]{mycolor}%
%     \usebeamercolor[fg]{navigation symbols}\insertslidenavigationsymbol%
%                                            \insertframenavigationsymbol%
%                                            \insertsubsectionnavigationsymbol%
%                                            \insertsectionnavigationsymbol%
%                                            \insertdocnavigationsymbol%
%                                            \insertbackfindforwardnavigationsymbol%
%     \end{beamercolorbox}%
%     }%
%     \vskip0.5pt%
% }{}

\mode<all>
