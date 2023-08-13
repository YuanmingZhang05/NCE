#include "utils.h"
#include <iostream>
#include <iomanip>
#include <string>
void load_graph(Graph& graph, std::string nodes_file, std::string edges_file, std::string groundtruth_file)
{
	std::ifstream fs_nodes(nodes_file);
	std::ifstream fs_edges(edges_file);
	std::ifstream fs_groundtruth(groundtruth_file);
	std::map<int, int> name_to_id;
	int n_nodes = 0; 
	vertex_info vertex;
	while (fs_nodes >> vertex.name >> vertex.weight)
	{
		boost::add_vertex(vertex, graph);
		name_to_id[vertex.name] = n_nodes; 
		n_nodes += 1;
	}
	int name; 
	float cardinality; 
	if (groundtruth_file != "")
	{
		while (fs_groundtruth >> name >> cardinality)
		{
			graph[name_to_id[name]].cardinality = cardinality;
		}
	}
	int from, to;
	while (fs_edges >> from >> to)
	{
		boost::add_edge(name_to_id[from], name_to_id[to], graph); 
	}
}