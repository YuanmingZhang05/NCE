#include "nce.h"
#include <iostream>
#include "hyperlogloglog/HyperLogLogLog.hpp"
uint32_t mmh3(uint64_t key, uint32_t seed)
{
	uint32_t hash; 
	MurmurHash3_x86_32(&key, 8, seed, &hash);
	return hash;
}
void nce_gc(Graph &graph, NCEParams params)
{
	int n_nodes = boost::num_vertices(graph), sketch_length = 1 << params.log2_length; 
	float* sketch_buffer = (float*)mkl_malloc((uint64_t)n_nodes * sketch_length * sizeof(float), 64), 
		*sketch_buffer_next = (float*)mkl_malloc(n_nodes * sketch_length * sizeof(float), 64);
	VSLStreamStatePtr random_stream; 
	vslNewStream(&random_stream, VSL_BRNG_SFMT19937, params.seed);
	vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, random_stream, n_nodes * sketch_length, sketch_buffer, 0.f, 1.f); 
	boost::graph_traits<Graph>::vertex_iterator current, end; 
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		cblas_sscal(sketch_length, 1. / graph[*current].weight, sketch_buffer + *current * sketch_length, 1); 
	}
	memcpy(sketch_buffer_next, sketch_buffer, n_nodes * sketch_length * sizeof(float));
	for (int hop = 0; hop < params.k; hop++)
	{
		boost::graph_traits<Graph>::vertex_iterator current, end;
		for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
		{
			boost::graph_traits<Graph>::adjacency_iterator current_neighbour, end_neighbour;
			for (boost::tie(current_neighbour, end_neighbour) = boost::adjacent_vertices(*current, graph); current_neighbour != end_neighbour; current_neighbour++)
			{
				vsFmin(sketch_length, sketch_buffer_next + *current * sketch_length,
					sketch_buffer + *current_neighbour * sketch_length, sketch_buffer_next + *current * sketch_length);
			}
		}
		memcpy(sketch_buffer, sketch_buffer_next, n_nodes * sketch_length * sizeof(float));
	}
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		graph[*current].estd_cardinality = (sketch_length - 1) / cblas_sasum(sketch_length, sketch_buffer + *current * sketch_length, 1); 
	}
	mkl_free(sketch_buffer);
	mkl_free(sketch_buffer_next);
}
void nce_lc(Graph& graph, NCEParams params)
{
	int n_nodes = boost::num_vertices(graph), sketch_length = 1 << params.log2_length, q = 1 << params.log2_q;
	uint8_t* sketch_buffer = (uint8_t*)mkl_calloc(1, n_nodes * sketch_length / 8, 64),
		* sketch_buffer_next = (uint8_t*)mkl_malloc(n_nodes * sketch_length / 8, 64);
	int* address_buffer = (int*)mkl_malloc(10000 * 4, 64);
	VSLStreamStatePtr random_stream;
	vslNewStream(&random_stream, VSL_BRNG_SFMT19937, params.seed);
	boost::graph_traits<Graph>::vertex_iterator current, end;
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		int n_p_nodes = int(q * graph[*current].weight);
		viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, random_stream, n_p_nodes, address_buffer, 0, sketch_length);
		for (uint32_t p_node = 0; p_node < n_p_nodes; p_node++)
		{
			sketch_buffer[*current * sketch_length / 8 + address_buffer[p_node] / 8] |= 1 << address_buffer[p_node] % 8;
		}
	}
	memcpy(sketch_buffer_next, sketch_buffer, n_nodes * sketch_length / 8);
	for (int hop = 0; hop < params.k; hop++)
	{
		boost::graph_traits<Graph>::vertex_iterator current, end;
		for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
		{
			boost::graph_traits<Graph>::adjacency_iterator current_neighbour, end_neighbour;
			for (boost::tie(current_neighbour, end_neighbour) = boost::adjacent_vertices(*current, graph); current_neighbour != end_neighbour; current_neighbour++)
			{
				for (int i = 0; i < sketch_length / 8; i++)
				{
					sketch_buffer_next[*current * sketch_length / 8 + i] |=
						sketch_buffer[*current_neighbour * sketch_length / 8 + i];
				}
				/*
				ippsOr_8u_I(
					sketch_buffer + *current_neighbour * sketch_length / 8, 
					sketch_buffer_next + *current * sketch_length / 8, 
					sketch_length / 8);
				*/
			}
		}
		memcpy(sketch_buffer, sketch_buffer_next, n_nodes * sketch_length / 8);
	}
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		int Z = 0;
		for (int i = 0; i < sketch_length / 8; i++)
		{
			Z += 8 - std::popcount(sketch_buffer[*current * sketch_length / 8 + i]);
		}
		graph[*current].estd_cardinality = -sketch_length * log(Z / (double)sketch_length) / q;
	}
	mkl_free(sketch_buffer);
	mkl_free(sketch_buffer_next);
	mkl_free(address_buffer);
}
void nce_hlll(Graph& graph, NCEParams params)
{
	int n_nodes = boost::num_vertices(graph), m = 1 << params.log2_length, q = 1 << params.log2_q;
	std::vector<hyperlogloglog::HyperLogLogLog<uint32_t>> sketch_buffer(n_nodes, hyperlogloglog::HyperLogLogLog<uint32_t>(m, 3)); 
	boost::graph_traits<Graph>::vertex_iterator current, end;
	VSLStreamStatePtr random_stream;
	vslNewStream(&random_stream, VSL_BRNG_SFMT19937, params.seed);
	uint32_t hash;
	int cnt = 0;
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		for (uint32_t p_node = 0; p_node < int(q * graph[*current].weight); p_node++)
		{
			viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, random_stream, 1, &hash); 
			sketch_buffer[*current].addHash(hash);
		}
		cnt++;
	}
	std::vector<hyperlogloglog::HyperLogLogLog<uint32_t>> sketch_buffer_next(sketch_buffer);
	for (int hop = 0; hop < params.k; hop++)
	{
		boost::graph_traits<Graph>::vertex_iterator current, end;
		for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
		{
			boost::graph_traits<Graph>::adjacency_iterator current_neighbour, end_neighbour;
			for (boost::tie(current_neighbour, end_neighbour) = boost::adjacent_vertices(*current, graph); current_neighbour != end_neighbour; current_neighbour++)
			{
				for (int i = 0; i < m; i++)
				{
					sketch_buffer_next[*current] = sketch_buffer_next[*current].merge(sketch_buffer[*current_neighbour]);
				}
			}
		}
		sketch_buffer = sketch_buffer_next; 
	}
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		graph[*current].estd_cardinality = sketch_buffer[*current].estimate();
	}
}
void hyperanf(Graph& graph, NCEParams params)
{
	int n_nodes = boost::num_vertices(graph), m = 1 << params.log2_length, q = 1 << params.log2_q;
	uint8_t* sketch_buffer = (uint8_t*)mkl_calloc(1, n_nodes * m, 64),
		* sketch_buffer_next = (uint8_t*)mkl_calloc(1, n_nodes * m, 64);
	boost::graph_traits<Graph>::vertex_iterator current, end;
	VSLStreamStatePtr random_stream;
	vslNewStream(&random_stream, VSL_BRNG_SFMT19937, params.seed);
	uint32_t hash, mask = (uint32_t)-1 >> params.log2_length; 
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		for (uint32_t p_node = 0; p_node < int(q * graph[*current].weight); p_node++)
		{
			viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, random_stream, 1, &hash);
			uint32_t addr = hash & (m - 1),
				rho = _lzcnt_u32(hash | (m - 1)) + 1;
			sketch_buffer[*current * m + addr] = 
				std::max(sketch_buffer[*current * m + addr], (uint8_t)std::min((uint32_t)1 << 5, rho)); 
		}
	}
	memcpy(sketch_buffer_next, sketch_buffer, n_nodes * m);
	for (int hop = 0; hop < params.k; hop++)
	{
		boost::graph_traits<Graph>::vertex_iterator current, end;
		for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
		{
			boost::graph_traits<Graph>::adjacency_iterator current_neighbour, end_neighbour;
			for (boost::tie(current_neighbour, end_neighbour) = boost::adjacent_vertices(*current, graph); current_neighbour != end_neighbour; current_neighbour++)
			{
				for (int i = 0; i < m; i++)
				{
					sketch_buffer_next[*current * m + i] =
						std::max(sketch_buffer_next[*current * m + i], 
							sketch_buffer[*current_neighbour * m + i]);
				}
			}
		}
		memcpy(sketch_buffer, sketch_buffer_next, n_nodes * m);
	}
	float alpha;
	switch (m)
	{
	case 16:
		alpha = 0.673;
		break;
	case 32:
		alpha = 0.697; 
		break;
	case 64:
		alpha = 0.709; 
		break;
	default:
		alpha = 0.7213 / (1 + 1.079 / m);
		break;
	}
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		float D = 0;
		for (int i = 0; i < m; i++)
		{
			D += pow(2, -sketch_buffer[*current * m + i]); 
		}
		float E = alpha * m * m / D;
		if (E < 2.5 * m)
		{
			int V = 0; 
			for (int i = 0; i < m; i++)
			{
				if (!sketch_buffer[*current * m + i]) 
					V += 1;
			}
			if (V)
				E = m * log(m / (double)V);
		}
		if (E > 1.43165e8)
		{
			E = -4.294967296e9 * log(1 - E / 4.294967296e9);
		}
		graph[*current].estd_cardinality = E / q;
	}
	mkl_free(sketch_buffer);
	mkl_free(sketch_buffer_next);
}
void bruteforce(Graph& graph, int k)
{
	int n_nodes = boost::num_vertices(graph);
	std::vector<std::set<int>> neighbourhood_buffer(n_nodes), neighbourhood_buffer_next;
	boost::graph_traits<Graph>::vertex_iterator current, end;
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		neighbourhood_buffer[*current].insert(int(*current));
	}
	neighbourhood_buffer_next = neighbourhood_buffer;
	for (int hop = 0; hop < k; hop++)
	{
		boost::graph_traits<Graph>::vertex_iterator current, end;
		for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
		{
			boost::graph_traits<Graph>::adjacency_iterator current_neighbour, end_neighbour;
			for (boost::tie(current_neighbour, end_neighbour) = boost::adjacent_vertices(*current, graph); current_neighbour != end_neighbour; current_neighbour++)
			{
				neighbourhood_buffer_next[*current].insert(
					neighbourhood_buffer[*current_neighbour].begin(), 
					neighbourhood_buffer[*current_neighbour].end()); 
			}
		}
		neighbourhood_buffer = neighbourhood_buffer_next; 
	}
	for (boost::tie(current, end) = boost::vertices(graph); current != end; current++)
	{
		float cardinality = 0.f; 
		for (auto& v : neighbourhood_buffer[*current])
		{
			cardinality += graph[v].weight; 
		}
		graph[*current].cardinality = cardinality;
	}
}