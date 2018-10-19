/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <math.h>

#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
//#include "boost/filesystem.hpp"

// Specifically built header files
#include "headers/read_write.h"
#include "headers/tensor_calc.h"
#include "headers/stmd_sync.h"

// Include of the FE model to solve in the simulation
// (hardcoded for the moment, but should try to split the solving and
// the mesh+BC in different headers)
//#include "headers/fe_problem_dropweight.h"
#include "headers/fe_problem_hopk.h"
//#include "headers/fe-spline_problem_hopk.h"

// To avoid conflicts...
// pointers.h in input.h defines MIN and MAX
// which are later redefined in petsc headers
#undef  MIN
#undef  MAX

#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/mpi.h>

namespace HMM
{
	using namespace dealii;

	template <int dim>
	class HMMProblem
	{
	public:
		HMMProblem ();
		~HMMProblem ();
		void run (std::string inputfile);

	private:
		void read_inputs(std::string inputfile);

		void set_global_communicators ();
		void set_repositories ();

		void do_timestep ();

		STMDSync<dim> 						*mmd_problem = NULL;
		FEProblem<dim> 						*fe_problem = NULL;

		MPI_Comm 							world_communicator;
		const int 							n_world_processes;
		const int 							this_world_process;
		int 								world_pcolor;

		MPI_Comm 							fe_communicator;
		int									root_fe_process;
		int 								n_fe_processes;
		int 								this_fe_process;
		int 								fe_pcolor;

		MPI_Comm 							mmd_communicator;
		int 								n_mmd_processes;
		int									root_mmd_process;
		int 								this_mmd_process;
		int 								mmd_pcolor;

		unsigned int						machine_ppn;
		int									fenodes;
		unsigned int						batch_nnodes_min;

		ConditionalOStream 					hcout;

		int									start_timestep;
		int									end_timestep;
		double              				present_time;
		double              				fe_timestep_length;
		double              				end_time;
		int        							timestep;
		int        							newtonstep;

		int									fe_degree;
		int									quadrature_formula;

		std::vector<std::string>			mdtype;
		unsigned int						nrepl;
		Tensor<1,dim> 						cg_dir;

		bool								activate_md_update;
		bool								use_pjm_scheduler;

		double								md_timestep_length;
		double								md_temperature;
		int									md_nsteps_sample;
		double								md_strain_rate;
		std::string							md_force_field;

		int									freq_checkpoint;
		int									freq_output_visu;
		int									freq_output_lhist;
		int									freq_output_homog;

		std::string                         macrostatelocin;
		std::string                         macrostatelocout;
		std::string							macrostatelocres;
		std::string							macrologloc;

		std::string                         nanostatelocin;
		std::string							nanostatelocout;
		std::string							nanostatelocres;
		std::string							nanologloc;
		std::string							nanologloctmp;
		std::string							nanologlochom;

		std::string							md_scripts_directory;


	};



	template <int dim>
	HMMProblem<dim>::HMMProblem ()
	:
		world_communicator (MPI_COMM_WORLD),
		n_world_processes (Utilities::MPI::n_mpi_processes(world_communicator)),
		this_world_process (Utilities::MPI::this_mpi_process(world_communicator)),
		world_pcolor (0),
		hcout (std::cout,(this_world_process == 0))
	{}



	template <int dim>
	HMMProblem<dim>::~HMMProblem ()
	{}




	template <int dim>
	void HMMProblem<dim>::read_inputs (std::string inputfile)
	{
	    using boost::property_tree::ptree;

	    std::ifstream jsonFile(inputfile);
	    ptree pt;
	    try{
		    read_json(jsonFile, pt);
	    }
	    catch (const boost::property_tree::json_parser::json_parser_error& e)
	    {
	        hcout << "Invalid JSON HMM input file (" << inputfile << ")" << std::endl;  // Never gets here
	    }
	    
            // Continuum timestepping
	    fe_timestep_length = std::stod(bptree_read(pt, "continuum time", "timestep length"));
	    start_timestep = std::stoi(bptree_read(pt, "continuum time", "start timestep"));
	    end_timestep = std::stoi(bptree_read(pt, "continuum time", "end timestep"));

	    // Continuum meshing
	    fe_degree = std::stoi(bptree_read(pt, "continuum mesh", "fe degree"));
	    quadrature_formula = std::stoi(bptree_read(pt, "continuum mesh", "quadrature formula"));

	    // Scale-bridging parameters
	    activate_md_update = std::stoi(bptree_read(pt, "scale-bridging", "activate md update"));
	    use_pjm_scheduler = std::stoi(bptree_read(pt, "scale-bridging", "use pjm scheduler"));

	    // Continuum input, output, restart and log location
		macrostatelocin = bptree_read(pt, "directory structure", "macroscale input");
		macrostatelocout = bptree_read(pt, "directory structure", "macroscale output");
		macrostatelocres = bptree_read(pt, "directory structure", "macroscale restart");
		macrologloc= bptree_read(pt, "directory structure", "macroscale log");

		// Atomic input, output, restart and log location
		nanostatelocin = bptree_read(pt, "directory structure", "nanoscale input");
		nanostatelocout = bptree_read(pt, "directory structure", "nanoscale output");
		nanostatelocres = bptree_read(pt, "directory structure", "nanoscale restart");
		nanologloc = bptree_read(pt, "directory structure", "nanoscale log");

		// Molecular dynamics material data
		nrepl = std::stoi(bptree_read(pt, "molecular dynamics material", "number of replicas"));
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
				get_subbptree(pt, "molecular dynamics material").get_child("list of materials.")) {
			mdtype.push_back(v.second.data());
		}
		// Direction to which all MD data are rotated to, to later ease rotation in the FE problem. The
		// replicas results are rotated to this referential before ensemble averaging, and the continuum
		// tensors are rotated to this referential from the microstructure given orientation
		std::vector<double> tmp_dir;
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
				get_subbptree(pt, "molecular dynamics material").get_child("rotation common ground vector.")) {
			tmp_dir.push_back(std::stod(v.second.data()));
		}
		if(tmp_dir.size()==dim){
			for(unsigned int imd=0; imd<dim; imd++){
				cg_dir[imd] = tmp_dir[imd];
			}
		}

		// Molecular dynamics simulation parameters
		md_timestep_length = std::stod(bptree_read(pt, "molecular dynamics parameters", "timestep length"));
		md_temperature = std::stod(bptree_read(pt, "molecular dynamics parameters", "temperature"));
		md_nsteps_sample = std::stoi(bptree_read(pt, "molecular dynamics parameters", "number of sampling steps"));
		md_strain_rate = std::stod(bptree_read(pt, "molecular dynamics parameters", "strain rate"));
		md_force_field = bptree_read(pt, "molecular dynamics parameters", "force field");
		md_scripts_directory = bptree_read(pt, "molecular dynamics parameters", "scripts directory");

		// Computational resources
		machine_ppn = std::stoi(bptree_read(pt, "computational resources", "machine cores per node"));
		fenodes = std::stoi(bptree_read(pt, "computational resources", "number of nodes for FEM simulation"));
		batch_nnodes_min = std::stoi(bptree_read(pt, "computational resources", "minimum nodes per MD simulation"));

		// Output and checkpointing frequencies
		freq_checkpoint = std::stoi(bptree_read(pt, "output data", "checkpoint frequency"));
		freq_output_lhist = std::stoi(bptree_read(pt, "output data", "visualisation output frequency"));
		freq_output_visu = std::stoi(bptree_read(pt, "output data", "analytics output frequency"));
		freq_output_homog = std::stoi(bptree_read(pt, "output data", "homogenization output frequency"));

		// Print a recap of all the parameters...
		hcout << "Parameters listing:" << std::endl;
		hcout << " - Activate MD updates (1 is true, 0 is false): "<< activate_md_update << std::endl;
		hcout << " - Use Pilot Job Manager to schedule MD jobs: "<< use_pjm_scheduler << std::endl;
		hcout << " - FE timestep duration: "<< fe_timestep_length << std::endl;
		hcout << " - Start timestep: "<< start_timestep << std::endl;
		hcout << " - End timestep: "<< end_timestep << std::endl;
		hcout << " - FE shape funciton degree: "<< fe_degree << std::endl;
		hcout << " - FE quadrature formula: "<< quadrature_formula << std::endl;
		hcout << " - Number of replicas: "<< nrepl << std::endl;
		hcout << " - List of material names: "<< std::flush;
		for(unsigned int imd=0; imd<mdtype.size(); imd++) hcout << " " << mdtype[imd] << std::flush; hcout << std::endl;;
		hcout << " - Direction use as a common ground/referential to transfer data between nano- and micro-structures : "<< std::flush;
		for(unsigned int imd=0; imd<dim; imd++) hcout << " " << cg_dir[imd] << std::flush; hcout << std::endl;;
		hcout << " - MD timestep duration: "<< md_timestep_length << std::endl;
		hcout << " - MD thermostat temperature: "<< md_temperature << std::endl;
		hcout << " - MD deformation rate: "<< md_strain_rate << std::endl;
		hcout << " - MD number of sampling steps: "<< md_nsteps_sample << std::endl;
		hcout << " - MD force field type: "<< md_force_field << std::endl;
		hcout << " - MD scripts directory (contains in.set, in.strain, ELASTIC/, ffield parameters): "<< md_scripts_directory << std::endl;
		hcout << " - Number of cores per node on the machine: "<< machine_ppn << std::endl;
		hcout << " - Number of nodes for FEM simulation: "<< fenodes << std::endl;
		hcout << " - Minimum number of nodes per MD simulation: "<< batch_nnodes_min << std::endl;
		hcout << " - Frequency of checkpointing: "<< freq_checkpoint << std::endl;
		hcout << " - Frequency of writing FE data files: "<< freq_output_lhist << std::endl;
		hcout << " - Frequency of writing FE visualisation files: "<< freq_output_visu << std::endl;
		hcout << " - Frequency of writing MD homogenization trajectory files: "<< freq_output_homog << std::endl;
		hcout << " - FE input directory: "<< macrostatelocin << std::endl;
		hcout << " - FE output directory: "<< macrostatelocout << std::endl;
		hcout << " - FE restart directory: "<< macrostatelocres << std::endl;
		hcout << " - FE log directory: "<< macrologloc << std::endl;
		hcout << " - MD input directory: "<< nanostatelocin << std::endl;
		hcout << " - MD output directory: "<< nanostatelocout << std::endl;
		hcout << " - MD restart directory: "<< nanostatelocres << std::endl;
		hcout << " - MD log directory: "<< nanologloc << std::endl;
	}




	template <int dim>
	void HMMProblem<dim>::set_global_communicators ()
	{
		//Setting up DEALII communicator and related variables
		root_fe_process = 0;
		n_fe_processes = fenodes*machine_ppn;
		// Color set above 0 for processors that are going to be used
		fe_pcolor = MPI_UNDEFINED;
		if (this_world_process >= root_fe_process &&
				this_world_process < root_fe_process + n_fe_processes) fe_pcolor = 0;
		else fe_pcolor = 1;

		MPI_Comm_split(world_communicator, fe_pcolor, this_world_process, &fe_communicator);
		MPI_Comm_rank(fe_communicator, &this_fe_process);


		//Setting up LAMMPS communicator and related variables
		root_mmd_process = 0;
		n_mmd_processes = n_world_processes;
		// Color set above 0 for processors that are going to be used
		mmd_pcolor = MPI_UNDEFINED;
		if (this_world_process >= root_mmd_process &&
				this_world_process < root_mmd_process + n_mmd_processes) mmd_pcolor = 0;
		else mmd_pcolor = 1;

		MPI_Comm_split(world_communicator, mmd_pcolor, this_world_process, &mmd_communicator);
		MPI_Comm_rank(mmd_communicator, &this_mmd_process);
	}




	template <int dim>
	void HMMProblem<dim>::set_repositories ()
	{
		if(!file_exists(macrostatelocin) || !file_exists(nanostatelocin)){
			std::cerr << "Missing macroscale or nanoscale input directories." << std::endl;
			exit(1);
		}

		mkdir(macrostatelocout.c_str(), ACCESSPERMS);
		mkdir(macrostatelocres.c_str(), ACCESSPERMS);
		mkdir(macrologloc.c_str(), ACCESSPERMS);

		mkdir(nanostatelocout.c_str(), ACCESSPERMS);
		mkdir(nanostatelocres.c_str(), ACCESSPERMS);
		mkdir(nanologloc.c_str(), ACCESSPERMS);
		nanologloctmp = nanologloc+"/tmp"; mkdir(nanologloctmp.c_str(), ACCESSPERMS);
		nanologlochom = nanologloc+"/homog"; mkdir(nanologlochom.c_str(), ACCESSPERMS);

		char fnset[1024]; sprintf(fnset, "%s/in.set.lammps", md_scripts_directory.c_str());
		char fnstrain[1024]; sprintf(fnstrain, "%s/in.strain.lammps", md_scripts_directory.c_str());
		char fnelastic[1024]; sprintf(fnelastic, "%s/ELASTIC", md_scripts_directory.c_str());

		if(!file_exists(fnset) || !file_exists(fnstrain) || !file_exists(fnelastic)){
			std::cerr << "Missing some MD input scripts for executing LAMMPS simulation (in 'box' directory)." << std::endl;
			exit(1);
		}
	}




	template <int dim>
	void HMMProblem<dim>::do_timestep ()
	{

		// Updating time variable
		present_time += fe_timestep_length;
		++timestep;
		hcout << "Timestep " << timestep << " at time " << present_time
				<< std::endl;
		if (present_time > end_time)
		{
			fe_timestep_length -= (present_time - end_time);
			present_time = end_time;
		}

		newtonstep = 0;

		// Initialisation of timestep variables
		if(fe_pcolor==0) fe_problem->beginstep(timestep, present_time);

		MPI_Barrier(world_communicator);

		// Solving iteratively the current timestep
		bool continue_newton = false;

		do
		{
			++newtonstep;

			if(fe_pcolor==0) fe_problem->solve(newtonstep);

			MPI_Barrier(world_communicator);

			if(mmd_pcolor==0) mmd_problem->update(timestep, present_time, newtonstep);
			MPI_Barrier(world_communicator);

			if(fe_pcolor==0) continue_newton = fe_problem->check();

			// Share the value of previous_res with processors outside of dealii allocation
			MPI_Bcast(&continue_newton, 1, MPI_C_BOOL, root_fe_process, world_communicator);

		} while (continue_newton);

		if(fe_pcolor==0) fe_problem->endstep();

		MPI_Barrier(world_communicator);
	}



	template <int dim>
	void HMMProblem<dim>::run (std::string inputfile)
	{
		// Reading JSON input file (common ones only)
		read_inputs(inputfile);

		hcout << "Building the HMM problem:       " << std::endl;
  	        // Set the dealii communicator using a limited amount of available processors
		// because dealii fails if processors do not have assigned cells. Plus, dealii
		// might not scale indefinitely
		set_global_communicators();

		// Setting repositories for input and creating repositories for outputs
		set_repositories();

		// Instantiation of the MMD Problem
		if(mmd_pcolor==0) mmd_problem = new STMDSync<dim> (mmd_communicator, mmd_pcolor);

		// Instantiation of the FE problem
		if(fe_pcolor==0) fe_problem = new FEProblem<dim> (fe_communicator, fe_pcolor, fe_degree, quadrature_formula);

		MPI_Barrier(world_communicator);

		// Initialization of time variables
		timestep = start_timestep - 1;
		present_time = timestep*fe_timestep_length;
		end_time = end_timestep*fe_timestep_length;

		hcout << " Initialization of the Multiple Molecular Dynamics problem...       " << std::endl;
		if(mmd_pcolor==0) mmd_problem->init(start_timestep, md_timestep_length, md_temperature,
											   md_nsteps_sample, md_strain_rate, md_force_field, nanostatelocin,
											   nanostatelocout, nanostatelocres, nanologloc,
											   nanologloctmp, nanologlochom, macrostatelocout,
											   md_scripts_directory, freq_checkpoint, freq_output_homog,
											   batch_nnodes_min, machine_ppn, mdtype, cg_dir, nrepl,
											   use_pjm_scheduler);

		// Initialization of MMD must be done before initialization of FE, because FE needs initial
		// materials properties obtained from MMD initialization
		MPI_Barrier(world_communicator);

		hcout << " Initiation of the Finite Element problem...       " << std::endl;
		if(fe_pcolor==0) fe_problem->init(start_timestep, fe_timestep_length,
										 macrostatelocin, macrostatelocout,
										 macrostatelocres, macrologloc,
										 freq_checkpoint, freq_output_visu, freq_output_lhist,
										 activate_md_update, mdtype, cg_dir);

		MPI_Barrier(world_communicator);

		// Running the solution algorithm of the FE problem
		hcout << "Beginning of incremental solution algorithm:       " << std::endl;
		while (present_time < end_time){
			do_timestep();
		}

		if(mmd_pcolor==0) delete mmd_problem;
		if(fe_pcolor==0) delete fe_problem;
	}
}



int main (int argc, char **argv)
{
	try
	{
		using namespace HMM;

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		if(argc!=2){
			std::cerr << "Wrong number of arguments, expected: './dealammps inputs_dealammps.json', but argc is " << argc << std::endl;
			exit(1);
		}

		std::string inputfile = argv[1];
		if(!file_exists(inputfile)){
			std::cerr << "Missing HMM input file." << std::endl;
			exit(1);
		}

		HMMProblem<3> hmm_problem;
		hmm_problem.run(inputfile);
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
