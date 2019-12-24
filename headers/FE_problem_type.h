#ifndef PROBLEM_TYPE_H
#define PROBLEM_TYPE_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>
#include <fstream>
#include <math.h>
#include <tuple>

#include "boost/property_tree/ptree.hpp"

//#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

namespace HMM {

struct MeshDimensions {
    double x, y, z;
    uint32_t x_cells, y_cells, z_cells;
};

template <int dim>
class ProblemType {
public:
    virtual void make_grid(parallel::shared::Triangulation<dim> &triangulation) = 0;
    virtual void define_boundary_conditions(DoFHandler<dim> &dof_handler) = 0;
    virtual std::map<types::global_dof_index, double> set_boundary_conditions(uint32_t timestep, double dt) = 0;
    virtual std::map<types::global_dof_index, double> boundary_conditions_to_zero(uint32_t timestep) = 0;

    MeshDimensions read_mesh_dimensions(boost::property_tree::ptree input_config) {
        MeshDimensions mesh;
        mesh.x = input_config.get<double>("continuum mesh.input.x length");
        mesh.y = input_config.get<double>("continuum mesh.input.y length");
        mesh.z = input_config.get<double>("continuum mesh.input.z length");
        mesh.x_cells = input_config.get<uint32_t>("continuum mesh.input.x cells");
        mesh.y_cells = input_config.get<uint32_t>("continuum mesh.input.y cells");
        mesh.z_cells = input_config.get<uint32_t>("continuum mesh.input.z cells");

        if (mesh.x < 0 || mesh.y < 0 || mesh.z < 0) {
            fprintf(stderr, "Mesh lengths must be positive \n");
            exit(1);
        }
        if (mesh.x_cells < 1 || mesh.y_cells < 1 || mesh.z_cells < 1 ) {
            fprintf(stderr, "Must be at least 1 cell per axis \n");
            exit(1);
        }
        return mesh;
    }
};
template <int dim>
class Dogbone: public ProblemType<dim> {
public:
    Dogbone (boost::property_tree::ptree input) {
        input_config = input;
        strain_rate = input_config.get<double>("problem type.strain rate");
    }

    void make_grid(parallel::shared::Triangulation<dim> &triangulation) {
        mesh = this->read_mesh_dimensions(input_config);
        // Generate block with bottom in plane 0,0. Strain applied in z axis
        Point<dim> corner1 (0, 0, 0);
        Point<dim> corner2 (mesh.x, mesh.y, mesh.z);

        std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells};
        GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
    }

    void define_boundary_conditions(DoFHandler<dim> &dof_handler) {
        typename DoFHandler<dim>::active_cell_iterator cell;
        unsigned int pointNumber = 0;
        for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
            double eps = cell->minimum_vertex_distance();
            double delta = eps / 10.0;
            for (uint32_t face = 0; face < GeometryInfo<3>::faces_per_cell; ++face) {
                for (uint32_t vert = 0; vert < GeometryInfo<3>::vertices_per_face; ++vert) {
                    pointNumber++;
                    // Point coords
                    double vertex_z = cell->face(face)->vertex(vert)(2);
                    // is vertex at base
                    if ( std::abs(vertex_z - 0.0) < delta ) {
                        for (uint32_t i = 0; i < dim; i++) {
                            fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, i) );
                        }
                    }

                    // is vertex on top
                    if ( std::abs(vertex_z - mesh.z) < delta) {
                        // fix in x,y; load along z axis
                        fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 0) );
                        fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 1) );
                        loaded_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 2) );
                        //std::cout << "LOAD " << cell->face(face)->vertex_dof_index(vert, 2) << std::endl;
                    }
                }
            }
        }
        std::sort( fixed_vertices.begin(), fixed_vertices.end() );
        fixed_vertices.erase( unique( fixed_vertices.begin(), fixed_vertices.end() ), fixed_vertices.end() );
        std::sort( loaded_vertices.begin(), loaded_vertices.end() );
        loaded_vertices.erase( unique( loaded_vertices.begin(), loaded_vertices.end() ), loaded_vertices.end() );
    }

    std::map<types::global_dof_index, double> set_boundary_conditions(uint32_t timestep, double dt) {
        // define accelerations of boundary verticies
        std::map<types::global_dof_index, double> boundary_values;
        types::global_dof_index vert;

        // fixed verticies have acceleration 0
        for (uint32_t i = 0; i < fixed_vertices.size(); i++) {
            vert = fixed_vertices[i];
            boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
        }
        // apply constant strain to top
        // need to pass FE solver the velocity increment
        // first step
        double acceleration;
        if (timestep == 1) {
            //acceleration = strain_rate * mesh.z / dt;
            acceleration = strain_rate * mesh.z / dt;
            //std::cout << "SET ACC " << acceleration << " "<< strain_rate * mesh.z << std::endl;
        } else {
            acceleration = 0;
        }
        for (uint32_t i = 0; i < loaded_vertices.size(); i++) {
            vert = loaded_vertices[i];
            std::map<types::global_dof_index, double>::iterator it = boundary_values.find(vert);
            if (it != boundary_values.end())
                it->second = acceleration;
            else
                boundary_values.insert( std::pair<types::global_dof_index, double> (vert, acceleration) );
        }
        return boundary_values;
    }

    std::map<types::global_dof_index, double> boundary_conditions_to_zero(uint32_t timestep) {
        std::map<types::global_dof_index, double> boundary_values;
        uint32_t vert;

        for (uint32_t i = 0; i < fixed_vertices.size(); i++) {
            vert = fixed_vertices[i];
            boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
        }

        for (uint32_t i = 0; i < loaded_vertices.size(); i++) {
            vert = loaded_vertices[i];
            boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
        }

        return boundary_values;
    }



private:
    boost::property_tree::ptree input_config;
    MeshDimensions                          mesh;

    std::vector<uint32_t>           fixed_vertices;
    std::vector<uint32_t>         loaded_vertices;

    double strain_rate;
};
template <int dim>
class DropWeight: public ProblemType<dim> {
public:
    DropWeight (boost::property_tree::ptree input) {
        input_config = input;
        n_accelerate_steps = input_config.get<double>("problem type.steps to accelerate");
        acceleration = input_config.get<double>("problem type.acceleration");
        timestep_length = input_config.get<double>("continuum time.timestep length");
        velocity_increment = -acceleration * timestep_length;
    }

    void make_grid(parallel::shared::Triangulation<dim> &triangulation) {
        mesh = this->read_mesh_dimensions(input_config);

        // Generate grid centred on 0,0 ; the top face is in plane with z=0
        Point<dim> corner1 (-mesh.x / 2, -mesh.y / 2, -mesh.z);
        Point<dim> corner2 (mesh.x / 2, mesh.y / 2, 0);

        std::vector<uint32_t> reps {mesh.x_cells, mesh.y_cells, mesh.z_cells};
        GridGenerator::subdivided_hyper_rectangle(triangulation, reps, corner1, corner2);
    }

    void define_boundary_conditions(DoFHandler<dim> &dof_handler) {
        double diam_weight = input_config.get<double>("problem type.diameter");

        typename DoFHandler<dim>::active_cell_iterator cell;

        for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
            double eps = cell->minimum_vertex_distance();
            for (uint32_t face = 0; face < GeometryInfo<3>::faces_per_cell; ++face) {
                for (uint32_t vert = 0; vert < GeometryInfo<3>::vertices_per_face; ++vert) {

                    // Point coords
                    double vertex_x = cell->face(face)->vertex(vert)(0);
                    double vertex_y = cell->face(face)->vertex(vert)(1);

                    // in plane distance between vertex and centre of dropweight, whihc is at 0,0
                    double x_dist = (vertex_x - 0.);
                    double y_dist = (vertex_y - 0.);
                    double dcwght = sqrt( x_dist * x_dist + y_dist * y_dist );

                    // is vertex impacted by the drop weight
                    if ((dcwght < diam_weight / 2.)) {
                        //for (uint32_t i=0; i<dim; i++){
                        loaded_vertices.push_back( cell->face(face)->vertex_dof_index(vert, 2) );
                        //}
                    }

                    // is point on the edge, if so it will be kept stationary
                    double delta = eps / 10.0; // in a grid, this will be small enough that only edges are used
                    if (   ( std::abs(vertex_x - mesh.x / 2) < delta )
                            || ( std::abs(vertex_x + mesh.x / 2) < delta )
                            || ( std::abs(vertex_y - mesh.y / 2) < delta )
                            || ( std::abs(vertex_y + mesh.y / 2) < delta ))

                        //if (   vertex_x > ( mesh.x/2 - delta)
                        //    || vertex_x < (-mesh.x/2 + delta)
                        //    || vertex_y > ( mesh.y/2 - delta)
                        //    || vertex_y < (-mesh.y/2 + delta))
                    {
                        for (uint32_t axis = 0; axis < dim; axis++) {
                            fixed_vertices.push_back( cell->face(face)->vertex_dof_index(vert, axis) );
                        }
                    }
                }
            }
        }
    }

    std::map<types::global_dof_index, double> set_boundary_conditions(uint32_t timestep, double dt) {
        // define accelerations of boundary verticies
        std::map<types::global_dof_index, double> boundary_values;
        types::global_dof_index vert;

        // fixed verticies have acceleration 0
        for (uint32_t i = 0; i < fixed_vertices.size(); i++) {
            vert = fixed_vertices[i];
            boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
        }

        // loaded verticies have const acceleration for first acc_steps
        if (timestep <= n_accelerate_steps) {
            for (uint32_t i = 0; i < loaded_vertices.size(); i++) {
                vert = loaded_vertices[i];
                boundary_values.insert( std::pair<types::global_dof_index, double> (vert, velocity_increment) );
            }
        }

        return boundary_values;
    }

    std::map<types::global_dof_index, double> boundary_conditions_to_zero(uint32_t timestep) {
        std::map<types::global_dof_index, double> boundary_values;
        uint32_t vert;

        for (uint32_t i = 0; i < fixed_vertices.size(); i++) {
            vert = fixed_vertices[i];
            boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
        }

        if (timestep <= n_accelerate_steps) {
            for (uint32_t i = 0; i < loaded_vertices.size(); i++) {
                vert = loaded_vertices[i];
                boundary_values.insert( std::pair<types::global_dof_index, double> (vert, 0.0) );
            }
        }

        return boundary_values;
    }



private:
    boost::property_tree::ptree input_config;
    MeshDimensions                          mesh;

    std::vector<uint32_t>           fixed_vertices;
    std::vector<uint32_t>         loaded_vertices;

    uint32_t    n_accelerate_steps;
    double      acceleration;
    double    timestep_length;
    double    velocity_increment;
};
}

#endif
