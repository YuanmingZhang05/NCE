#pragma once
#include <mkl.h>
#include <numeric>
#include <bit>
#include <bitset>
#include "utils.h"
#include "mmh3.h"
#define METHOD_BRUTEFORCE 0
#define METHOD_GC 1
#define METHOD_HYPERANF 2
#define METHOD_LC 3
typedef struct
{
	int k, log2_q, log2_length, n_rounds;
	unsigned seed;
}NCEParams;
void nce_gc(Graph& graph, NCEParams params); 
void nce_lc(Graph& graph, NCEParams params);
void hyperanf(Graph& graph, NCEParams params);
void nce_hlll(Graph& graph, NCEParams params); 
void bruteforce(Graph& graph, int k); 