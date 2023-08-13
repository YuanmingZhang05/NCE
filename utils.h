#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
typedef struct
{
	int name; 
	float weight, cardinality, estd_cardinality;
}vertex_info;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_info> Graph;
void load_graph(Graph& graph, std::string nodes_file, std::string edges_file, std::string groundtruth_file = "");