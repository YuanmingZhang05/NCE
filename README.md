# Acceleration with SIMD Instruction
In the following, we introduce how to accelerate sketch recursive aggregation.
The sketch aggregation process consists of intensive yet simple comparison operations between GC sketches from each node's neighborhood.
However, comparison operations of different registers $S^{(j)}[i]$ are independent. 
Therefore, we use SIMD (single instruction multiple data) instructions to maximize processors usage 
by parallel processing comparison operations between different registers. 
Many modern processors, such as Intel and AMD, support the SIMD instructions to split a SIMD register ($e.g.$, 128-bit, 256-bit, 512-bit) in the processor into several low-bit ($e.g.$, 8-bit, 32-bit, 64-bit) registers,
and these low-bit registers work parallel. 
Hence, we assign the comparison operations in a register $S_v[i]$ to a low-bit register.
Specifically, we use a $32$-bit floating-point register to store an element $S_v[i]$ in the GC sketch.
For any two GC sketch $S_v$ and $S_u$ of length $m$, 
we use the following SIMD instruction (refer to the [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide) for more details), 

```math
\mathtt{\_mm\_min\_ps\; (\_\;\!\_m128\; \textit{a}\;\!,\; \_\;\!\_m128\; \textit{b})},
```
as a sliding window to parallel compute 4 continuous elements in sketches, $i.e.$,

$$ a,b \gets (S[4i+1],S[4i+2],S[4i+3],S[4i+4]), \quad 0 \leq i \leq \left\lceil \frac{m}{4} \right\rceil,$$

where SIMD registers $a$ and $b$ get elements from $S_v$ and $S_u$ respectively.
Then, under a 128-bit register, 
the above SIMD instruction parallel computes $\min(a[i],b[i])$, $1\leq i \leq 4$.
Also, for 256-bit and 512-bit registers Intel Advanced Vector Extensions (Intel AVX) and AVX-512 provide intrinsics:

```math
\begin{split}
&\mathtt{\_mm256\_min\_ps (\_\_m256 \textit{a}, \_\_m256 \textit{b})},\\
&\mathtt{\_mm512\_min\_ps (\_\_m512 \textit{a}, \_\_m512 \textit{b})},
\end{split}
```

which can parallel process 8 and 16 elements of our GC sketches in a register respectively.
One can use SIMD instructions through both assembly and C/C++ codes. 
To facilitate using SIMD instructions for our GC sketch,
we provide C intrinsic functions for loading/writing continuous elements in sketches to/from a SIMD register.
In detail, for a 128-, 256-, and 512-bit SIMD register we use the following instructions


```math
\begin{split}
&\mathtt{\_mm\_load\_ps\; (float\;\; const\; ^*mem\_addr)}\\
&\mathtt{\_mm256\_load\_ps\; (float\;\; const\; ^*mem\_addr)}\\
&\mathtt{\_mm512\_load\_ps\; (void\;\; const\;\; ^*mem\_addr)}
\end{split}
```

to obtain 4, 8, and 16 elements of GC sketches from 16-, 32-, and 64-byte-aligned memory addresses $\mathtt{mem\_addr}$, respectively.
In addition, we use the following instructions


```math
\begin{split}
&\mathtt{\_mm\_store\_ps\; (float\;\;^*mem\_addr\;\!,\; \_\;\!\_m128\; \textit{a})}\\
&\mathtt{\_mm256\_store\_ps\; (float\;\;^*mem\_addr\;\!,\; \_\;\!\_m256\; \textit{a})}\\
&\mathtt{\_mm512\_store\_ps\; (void\;\;^*mem\_addr\;\!,\; \_\;\!\_m512\;\; \textit{a})}
\end{split}
```

to write the after-processed elements in a 128-, 256-, and 512-bit SIMD register $a$ back to aligned memory, respectively.


Notice that the LC-NCE sketch consists of $m$ bits.
To speed up the merge operation
between two LC-NCE sketches, which includes a large number of bitwise OR operations,
we use the following SIMD instruction
```math
\mathtt{\_\;\!\_m512i\;\; \_mm512\_or\_epi64\; (\_\;\!\_m512i\; \textit{a}\;\!,\; \_\;\!\_m512i\; \textit{b})},
```

to compute the bitwise OR of two 512-bit SIMD registers, each consisting of 8 packed 64-bit integers.
As for the estimation stage, we compute the number of logical 1 bits in $S^{(k)}_v$, denoted as $O_v$, using the following instruction 

```math
\mathtt{\_\;\!\_m512i\;\; \_mm512\_popcnt\_epi64\; (\_\;\!\_m512i\; \textit{a}}),
```
which counts logical 1 bits in 512-bit SIMD register $a$ consisting of packed 64-bit integers.
Then, we easily obtain $Z_v$ in Eq. (11) by subtracting $O_v$ from $m$. 


Similar to the GC sketch, the aggregation stage of HyperANF sketches involves intensive comparison operations between sketches.
The basic idea behind using SIMD instructions to speed up the sketch recursive aggregation is the same.
The only difference is that the elements in the GC sketch and HyperANF sketch are of different types which are floating-point numbers and unsigned integers, respectively.
Specifically, we adopt the following SIMD instruction
```math
\mathtt{\_mm\_min\_epu8\; (\_\;\!\_m512i\; \textit{a}\;\!,\; \_\;\!\_m512i\; \textit{b})},
```
to compare two integer arrays $a$ and $b$, each consisting of 64 unsigned 8-bit integers.
