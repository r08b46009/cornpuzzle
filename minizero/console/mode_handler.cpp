#include "mode_handler.h"
#include "actor_group.h"
#include "color_message.h"
#include "console.h"
#include "create_actor.h"
#include "create_network.h"
#include "git_info.h"
#include "obs_recover.h"
#include "obs_remover.h"
#include "ostream_redirector.h"
#include "random.h"
#include "zero_server.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace minizero::console {

using namespace minizero::utils;

namespace {

std::vector<std::filesystem::path> collectCornPuzzleFiles(const std::filesystem::path& puzzle_dir)
{
    std::vector<std::filesystem::path> puzzle_files;
    for (const auto& entry : std::filesystem::directory_iterator(puzzle_dir)) {
        if (!entry.is_regular_file()) { continue; }
        if (entry.path().extension() != ".txt") { continue; }
        puzzle_files.push_back(entry.path());
    }
    std::sort(puzzle_files.begin(), puzzle_files.end());
    return puzzle_files;
}

std::string sanitizeFileStem(std::string stem)
{
    for (char& c : stem) {
        if (c == '/' || c == '\\' || c == ':' || c == ' ' || c == '\t') { c = '_'; }
    }
    return stem;
}

} // namespace

ModeHandler::ModeHandler()
{
    RegisterFunction("console", this, &ModeHandler::runConsole);
    RegisterFunction("sp", this, &ModeHandler::runSelfPlay);
    RegisterFunction("zero_server", this, &ModeHandler::runZeroServer);
    RegisterFunction("zero_training_name", this, &ModeHandler::runZeroTrainingName);
    RegisterFunction("env_test", this, &ModeHandler::runEnvTest);
    RegisterFunction("remove_obs", this, &ModeHandler::runRemoveObs);
    RegisterFunction("recover_obs", this, &ModeHandler::runRecoverObs);
    RegisterFunction("solve_cornpuzzle", this, &ModeHandler::runSolveCornPuzzle);
}

void ModeHandler::run(int argc, char* argv[])
{
    if (argc % 2 != 1) { usage(); }

    env::setUpEnv();

    std::string mode_string = "console";
    std::string config_file = "";
    std::string config_string = "";
    config::ConfigureLoader cl;
    setDefaultConfiguration(cl);

    std::string gen_config = "";
    for (int i = 1; i < argc; i += 2) {
        std::string sCommand = std::string(argv[i]);

        if (sCommand == "-mode") {
            mode_string = argv[i + 1];
        } else if (sCommand == "-gen") {
            gen_config = argv[i + 1];
        } else if (sCommand == "-conf_file") {
            config_file = argv[i + 1];
        } else if (sCommand == "-conf_str") {
            config_string = argv[i + 1];
        } else {
            std::cerr << "unknown argument: " << sCommand << std::endl;
            usage();
        }
    }

    if (!readConfiguration(cl, config_file, config_string)) { exit(-1); }
    utils::setColorOutputEnabled(config::program_use_color_message);                                      // set color message output
    utils::OstreamRedirector::silence(std::cerr, config::program_quiet);                                  // silence std::cerr if program_quiet
    utils::Random::seed(config::program_auto_seed ? static_cast<int>(time(NULL)) : config::program_seed); // setup random seed

    if (!gen_config.empty()) {
        // generate configuration file after reading cfg file
        genConfiguration(cl, gen_config);
        exit(0);
    } else {
        std::cerr << "(Version: " << GIT_SHORT_HASH << ")" << std::endl;
        // run mode
        if (!function_map_.count(mode_string)) { usage(); }
        (*function_map_[mode_string])();
    }
}

void ModeHandler::usage()
{
    std::cout << "./minizero [arguments]" << std::endl;
    std::cout << "arguments:" << std::endl;
    std::cout << "\t-mode [" << getAllModesString() << "]" << std::endl;
    std::cout << "\t-gen configuration_file" << std::endl;
    std::cout << "\t-conf_file configuration_file" << std::endl;
    std::cout << "\t-conf_str configuration_string" << std::endl;
    exit(-1);
}

std::string ModeHandler::getAllModesString()
{
    std::string mode_string;
    for (const auto& m : function_map_) { mode_string += (mode_string.empty() ? "" : "|") + m.first; }
    return mode_string;
}

void ModeHandler::genConfiguration(config::ConfigureLoader& cl, const std::string& config_file)
{
    // check configure file is exist
    std::ifstream f(config_file);
    if (f.good()) {
        char ans = ' ';
        while (ans != 'y' && ans != 'n') {
            std::cerr << config_file << " already exist, do you want to overwrite it? [y/n]" << std::endl;
            std::cin >> ans;
        }
        if (ans == 'y') { std::cerr << "overwrite " << config_file << std::endl; }
        if (ans == 'n') {
            std::cerr << "didn't overwrite " << config_file << std::endl;
            f.close();
            return;
        }
    }
    f.close();

    std::ofstream fout(config_file);
    fout << cl.toString();
    fout.close();
}

bool ModeHandler::readConfiguration(config::ConfigureLoader& cl, const std::string& config_file, const std::string& config_string)
{
    if (!config_file.empty() && !cl.loadFromFile(config_file)) {
        std::cerr << "Failed to load configuration file." << std::endl;
        return false;
    }
    if (!config_string.empty() && !cl.loadFromString(config_string)) {
        std::cerr << "Failed to load configuration string." << std::endl;
        return false;
    }

    if (!config::program_quiet) { std::cerr << cl.toString(); }
    return true;
}

void ModeHandler::runConsole()
{
    console::Console console;
    std::string command;
    console.initialize();
    std::cerr << "Successfully started console mode" << std::endl;
    while (getline(std::cin, command)) {
        if (command == "quit") { break; }
        console.executeCommand(command);
    }
}

void ModeHandler::runSelfPlay()
{
    actor::ActorGroup ag;
    ag.run();
}

void ModeHandler::runZeroServer()
{
    zero::ZeroServer server;
    server.run();
}

void ModeHandler::runZeroTrainingName()
{
    std::cout << Environment().name()                                                           // name for environment
              << "_" << (config::actor_use_gumbel ? "g" : "") << config::nn_type_name[0] << "z" // network & training algorithm
              << "_" << config::nn_num_blocks << "b"                                            // number of blocks
              << "x" << config::nn_num_hidden_channels                                          // number of hidden channels
              << "_n" << config::actor_num_simulation                                           // number of simulations
              << "-" << GIT_SHORT_HASH << std::endl;                                            // git hash info
}

void ModeHandler::runEnvTest()
{
    Environment env;
    env.reset();
    while (!env.isTerminal()) {
        std::vector<Action> legal_actions = env.getLegalActions();
        int index = utils::Random::randInt() % legal_actions.size();
        bool legal = env.isLegalAction(legal_actions[index]);
        bool success = env.act(legal_actions[index]);
        if (!legal || !success) { assert(false); }
    }
    std::cout << env.toString() << std::endl;

    EnvironmentLoader env_loader;
    env_loader.loadFromEnvironment(env);
    std::cout << env_loader.toString() << std::endl;

    std::string env_str = env.toString();
    env.reset();
    for (const auto& action_pair : env_loader.getActionPairs()) {
        bool legal = env.isLegalAction(action_pair.first);
        bool success = env.act(action_pair.first);
        if (!legal || !success) { assert(false); }
    }
    assert(env.toString() == env_str);
}

void ModeHandler::runRemoveObs()
{
    std::string obs_file_path;
    std::cin >> obs_file_path;

    minizero::env::atari::ObsRemover ob;
    ob.initialize();
    ob.run(obs_file_path);
}

void ModeHandler::runRecoverObs()
{
    std::string obs_file_path;
    std::cin >> obs_file_path;

#if ATARI
    minizero::env::atari::ObsRecover ob;
    ob.initialize();
    ob.run(obs_file_path);
#else
    std::cout << "Currently, only support recover observation for atari games" << std::endl;
#endif
}

void ModeHandler::runSolveCornPuzzle()
{
    const std::string puzzle_dir = minizero::config::env_compound_puzzles_dir;
    if (puzzle_dir.empty()) {
        std::cerr << "[solve_cornpuzzle] env_compound_puzzles_dir is empty" << std::endl;
        return;
    }

    std::filesystem::path root_dir(puzzle_dir);
    if (!std::filesystem::exists(root_dir) || !std::filesystem::is_directory(root_dir)) {
        std::cerr << "[solve_cornpuzzle] puzzle directory does not exist: " << root_dir << std::endl;
        return;
    }

    std::filesystem::path output_dir = config::test_output_path.empty()
                                           ? root_dir / "_solved"
                                           : std::filesystem::path(config::test_output_path);
    std::filesystem::path stdout_dir = output_dir / "stdout";
    std::filesystem::path sgf_dir = output_dir / "sgf";

    std::filesystem::create_directories(stdout_dir);
    std::filesystem::create_directories(sgf_dir);

    const std::vector<std::filesystem::path> puzzle_files = collectCornPuzzleFiles(root_dir);
    if (puzzle_files.empty()) {
        std::cerr << "[solve_cornpuzzle] no .txt puzzle files found in directory: " << root_dir << std::endl;
        return;
    }

    const float original_disable_resign_ratio = config::zero_disable_resign_ratio;
    config::zero_disable_resign_ratio = 1.0f;

    auto network = minizero::network::createNetwork(config::nn_file_name, 0);
    auto actor = minizero::actor::createActor(static_cast<uint64_t>(config::actor_num_simulation + 1) * network->getActionSize(), network);

    std::cout << "[solve_cornpuzzle] solving " << puzzle_files.size()
              << " puzzles from " << root_dir << std::endl;
    std::cout << "[solve_cornpuzzle] stdout: " << stdout_dir << std::endl;
    std::cout << "[solve_cornpuzzle] sgf: " << sgf_dir << std::endl;

    for (const auto& puzzle_file : puzzle_files) {
        const std::string puzzle_name = puzzle_file.filename().string();
        const std::string stem = sanitizeFileStem(puzzle_file.stem().string());

        actor->reset();
        actor->getEnvironment().resetFromPuzzleFile(puzzle_file.string());
        actor->resetSearch();
        actor->getActionInfoHistory().clear();

        std::ostringstream stdout_oss;
        stdout_oss << "[Puzzle] " << puzzle_file.string() << std::endl;
        stdout_oss << "[Initial]" << std::endl;
        stdout_oss << actor->getEnvironment().toString() << std::endl;

        int step = 0;
        while (!actor->isEnvTerminal()) {
            const auto legal_actions = actor->getEnvironment().getLegalActions();
            if (legal_actions.empty()) {
                stdout_oss << "[NoLegalAction]" << std::endl;
                break;
            }

            const Action action = actor->think(false, false);

            if (actor->isResign()) {
                stdout_oss << "step " << step << ": Resign" << std::endl;
                break;
            }

            const bool played = actor->act(action);
            stdout_oss << "step " << step++ << ": " << action.toConsoleString() << std::endl;
            if (!played) {
                stdout_oss << "[InvalidAction] " << action.toConsoleString() << std::endl;
                break;
            }
            stdout_oss << actor->getEnvironment().toString() << std::endl;
        }

        stdout_oss << "[Terminal] " << (actor->isEnvTerminal() ? "true" : "false") << std::endl;
        stdout_oss << "[EvalScore] " << actor->getEvalScore() << std::endl;
        stdout_oss << actor->getEnvironment().toString() << std::endl;

        const std::string stdout_path = (stdout_dir / (stem + ".txt")).string();
        std::ofstream stdout_file(stdout_path, std::ios::out | std::ios::trunc);
        if (!stdout_file) {
            std::cerr << "[solve_cornpuzzle] failed to write stdout file: " << stdout_path << std::endl;
            continue;
        }
        stdout_file << stdout_oss.str();
        stdout_file.close();

        const std::string sgf_path = (sgf_dir / (stem + ".sgf")).string();
        std::ofstream sgf_file(sgf_path, std::ios::out | std::ios::trunc);
        if (!sgf_file) {
            std::cerr << "[solve_cornpuzzle] failed to write sgf file: " << sgf_path << std::endl;
            continue;
        }
        sgf_file << actor->getRecord({{"SOURCE", "cornpuzzle"}, {"PUZZLE", puzzle_file.string()}}) << std::endl;
        sgf_file.close();

        std::cout << "[solve_cornpuzzle] " << puzzle_name << " -> " << stdout_path << " | " << sgf_path << std::endl;
    }

    config::zero_disable_resign_ratio = original_disable_resign_ratio;
}

} // namespace minizero::console
