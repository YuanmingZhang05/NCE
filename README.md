# NCE
in the following, we introduce how to accelerate sketch recursive aggregation.
The sketch aggregation process consists of intensive yet simple comparison operations between GC sketches from each node's neighborhood.
However, as shown in Eq.~(\ref{eq:aggregation}), comparison operations of different registers $S^{(j)}[i]$ are independent. 
Therefore, we use SIMD (single instruction multiple data) instructions to maximize processors usage 
by parallel processing comparison operations between different registers. 
Many modern processors, such as Intel and AMD, support the SIMD instructions to split a SIMD register (e.g., 128-bit, 256-bit, 512-bit) in the processor into several low-bit (e.g., 8-bit, 32-bit, 64-bit) registers,
and these low-bit registers work parallel. 
Hence, we assign the comparison operations in a register $S_v[i]$ to a low-bit register.
Specifically, we use a $32$-bit floating-point register to store an element $S_v[i]$ in the GC sketch.
% We use the following SIMD instructions to accelerate %the process of 
% the sketch recursive aggregation.
For any two GC sketch $S_v$ and $S_u$ of length $m$, 
we use the following SIMD instruction (refer to the Intel Intrinsics Guide\footnote{https://software.intel.com/sites/landingpage/IntrinsicsGuide} for more details), 

$$\tt{\_mm\_min\_ps\; (\_\;\!\_m128\; \emph{a}\;\!,\; \_\;\!\_m128\; \emph{b})},$$

%https://learn.microsoft.com/zh-cn/previous-versions/bb514097(v=vs.110)

as a sliding window to parallel compute 4 continuous elements in sketches, i.e.,

$$ a,b \gets (S[4i+1],S[4i+2],S[4i+3],S[4i+4]), \quad 0 \leq i \leq \left\lceil \frac{m}{4} \right\rceil,$$

where SIMD registers $a$ and $b$ get elements from $S_v$ and $S_u$ respectively.
Then, under a 128-bit register, 
the above SIMD instruction parallel computes $\min(a[i],b[i])$, $1\leq i \leq 4$.
Also, for 256-bit and 512-bit registers Intel Advanced Vector Extensions (Intel AVX) and AVX-512 provide intrinsics:
%To further speedup the procedure, Intel Advanced Vector Extensions (Intel AVX) and AVX-512 provide intrinsics:

$$\begin{cases}
&\mathtt{\_mm256\_min\_ps\; (\_\;\!\_m256\; \emph{a}\;\!,\; \_\;\!\_m256\; \emph{b})},\\
&\mathtt{\_mm512\_min\_ps\; (\_\;\!\_m512\; \emph{a}\;\!,\; \_\;\!\_m512\; \emph{b})},
\end{cases}$$

which can parallel process 8 and 16 elements of our GC sketches in a register respectively.
One can use SIMD instructions through both assembly and C/C++ codes. 
To facilitate using SIMD instructions for our GC sketch,
we provide C intrinsic functions for loading/writing continuous elements in sketches to/from a SIMD register.
In detail, for a 128-, 256-, and 512-bit SIMD register we use the following instructions

$$\begin{cases}
&\mathtt{\_mm\_load\_ps\; (float\;\; const\; ^*mem\_addr)}\\
&\mathtt{\_mm256\_load\_ps\; (float\;\; const\; ^*mem\_addr)}\\
&\mathtt{\_mm512\_load\_ps\; (void\;\; const\;\; ^*mem\_addr)}
\end{cases}$$

to obtain 4, 8, and 16 elements of GC sketches from 16-, 32-, and 64-byte-aligned memory addresses $\mathtt{mem\_addr}$, respectively.
In addition, we use the following instructions

$$\begin{cases}
&\mathtt{\_mm\_store\_ps\; (float\;\;^*mem\_addr\;\!,\; \_\;\!\_m128\; \emph{a})}\\
&\mathtt{\_mm256\_store\_ps\; (float\;\;^*mem\_addr\;\!,\; \_\;\!\_m256\; \emph{a})}\\
&\mathtt{\_mm512\_store\_ps\; (void\;\;^*mem\_addr\;\!,\; \_\;\!\_m512\;\; \emph{a})}
\end{cases}$$

to write the after-processed elements in a 128-, 256-, and 512-bit SIMD register $a$ back to aligned memory, respectively.
