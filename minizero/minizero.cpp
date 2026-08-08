#include "mode_handler.h"
#include "environment/cornpuzzle/cornpuzzle.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

namespace {

int runCornPuzzleRandomBaseline(int argc, char* argv[])
{
    int num_games = 100;
    unsigned int seed = 123;

    if (argc >= 3) {
        num_games = std::max(1, std::atoi(argv[2]));
    }

    if (argc >= 4) {
        seed = static_cast<unsigned int>(std::atoi(argv[3]));
    }

    std::mt19937 rng(seed);

    using namespace minizero::env::cornpuzzle;

    // This loads CORNPUZZLE_FILE and initializes all placement actions.
    CornPuzzleEnv::setUpEnv();

    double total_return = 0.0;
    double min_return = std::numeric_limits<double>::infinity();
    double max_return = -std::numeric_limits<double>::infinity();

    int solved_count = 0;
    int dead_end_count = 0;
    int total_steps = 0;
    int illegal_act_count = 0;

    std::cout << "===== CornPuzzle Random Baseline =====\n";
    std::cout << "Games: " << num_games << "\n";
    std::cout << "Seed: " << seed << "\n";

    for (int g = 0; g < num_games; ++g) {
        CornPuzzleEnv env;
        env.reset();

        int steps = 0;
        bool broken = false;

        while (!env.isTerminal()) {
            auto legal_actions = env.getLegalActions();

            if (legal_actions.empty()) {
                break;
            }

            std::uniform_int_distribution<int> dist(
                0,
                static_cast<int>(legal_actions.size()) - 1
            );

            const auto action = legal_actions[dist(rng)];

            bool ok = env.act(action);

            if (!ok) {
                illegal_act_count++;
                broken = true;

                std::cerr << "[Baseline error] act() returned false for legal action: "
                          << action.toConsoleString()
                          << "\n";
                break;
            }

            steps++;

            if (steps > 10000) {
                broken = true;
                std::cerr << "[Baseline warning] Game exceeded 10000 steps.\n";
                break;
            }
        }

        double final_return = env.getEvalScore();

        // In your bounded reward setting, solved states should reach 1.0.
        bool solved = (final_return >= 0.999);

        if (solved) {
            solved_count++;
        } else {
            dead_end_count++;
        }

        total_return += final_return;
        min_return = std::min(min_return, final_return);
        max_return = std::max(max_return, final_return);
        total_steps += steps;

        if (g < 10) {
            std::cout << "[Game " << g
                      << "] return=" << final_return
                      << ", solved=" << solved
                      << ", steps=" << steps
                      << ", terminal=" << env.isTerminal()
                      << ", broken=" << broken
                      << "\n";
        }
    }

    std::cout << "\n===== Summary =====\n";
    std::cout << "Avg return: "
              << total_return / static_cast<double>(num_games)
              << "\n";

    std::cout << "Min return: " << min_return << "\n";
    std::cout << "Max return: " << max_return << "\n";

    std::cout << "Solve rate: "
              << static_cast<double>(solved_count) / static_cast<double>(num_games)
              << "\n";

    std::cout << "Dead-end rate: "
              << static_cast<double>(dead_end_count) / static_cast<double>(num_games)
              << "\n";

    std::cout << "Avg steps: "
              << static_cast<double>(total_steps) / static_cast<double>(num_games)
              << "\n";

    std::cout << "Illegal act count: "
              << illegal_act_count
              << "\n";

    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc >= 2 && std::strcmp(argv[1], "cornpuzzle_random_baseline") == 0) {
        return runCornPuzzleRandomBaseline(argc, argv);
    }

    minizero::console::ModeHandler mode_handler;
    mode_handler.run(argc, argv);
    return 0;
}