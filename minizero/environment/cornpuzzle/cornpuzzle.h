#pragma once

#include "base_env.h"
#include <array>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

namespace minizero::env::cornpuzzle {

const std::string kCornPuzzleName = "cornpuzzle";
const int kCornPuzzleNumPlayer = 1;

// The model still sees a fixed 14 x 14 feature board.
// Only the first kCornPuzzlePlayableRows rows are used by the game.
// Curriculum / txt puzzles can activate a smaller active_rows x active_cols
// region inside this fixed board.
const int kCornPuzzlePlayableRows = 7;
const int kCornPuzzleRows = 14;
const int kCornPuzzleCols = 14;
const int kCornPuzzleBoardSize = kCornPuzzleRows * kCornPuzzleCols;
const int kCornPuzzlePlayableArea = kCornPuzzlePlayableRows * kCornPuzzleCols;

// Candidate-action version.
//
// kCornPuzzleNumPieces is now the maximum number of candidate shape slots
// in a single puzzle, not a fixed global piece library size.
// Each puzzle normalizes and groups its own pieces into candidate slots
// 0..num_candidates-1. Unused candidate slots are masked by legality.
const int kCornPuzzleMaxCandidateShapes = 64;
const int kCornPuzzleNumRotations = 2;  // currently supports 0 and 180 degrees
const int kCornPuzzleNumPieces = kCornPuzzleMaxCandidateShapes;

// Normalize the absolute remaining count used in candidate shape channels.
// Feature value = min(remaining_count / kCornPuzzleRemainingCountNorm, 1.0).
// This preserves count information better than remaining/initial ratio while
// keeping input values in roughly the same 0..1 scale as board channels.
const int kCornPuzzleRemainingCountNorm = 16;

// Candidate shape feature metadata. The current C++ action feature is still
// returned as a board-sized mask via getActionFeatures(), so pieces are not
// restricted to this box. These constants are kept for the future Python
// candidate-shape encoder.
const int kCornPuzzleShapeBoxRows = 4;
const int kCornPuzzleShapeBoxCols = 4;

const int kCornPuzzleDiscreteValueSize = 1;

enum class CornActionPhase {
    kSelectPiece,
    kSelectPosition,
};

struct CornPlacement {
    int piece_id;
    int rotation;
    int top;
    int left;
    std::vector<std::pair<int, int>> cells;
};

struct CornCandidateShape {
    int candidate_id = -1;
    int count = 0;
    int area = 0;
    int height = 0;
    int width = 0;
    std::string shape_key;
    std::vector<std::pair<int, int>> cells;  // normalized canonical cells
};

extern std::vector<CornPlacement> kCornPuzzleActions;
extern std::vector<std::string> kCornPuzzleActionName;
extern std::unordered_map<std::string, int> kCornPuzzleActionNameToID;

void initialize();

class CornPuzzleAction : public BaseAction {
public:
    CornPuzzleAction() : BaseAction() {}
    CornPuzzleAction(int action_id, Player player) : BaseAction(action_id, player) {}

    CornPuzzleAction(const std::vector<std::string>& action_string_args)
    {
        action_id_ = -1;
        player_ = Player::kPlayerNone;

        if (action_string_args.empty()) {
            return;
        }

        std::string token;
        if (action_string_args.size() >= 2) {
            token = action_string_args[1];
        } else {
            token = action_string_args[0];
        }

        // SGF stores actions as numeric IDs, e.g. B[41], B[854].
        bool is_number = !token.empty() &&
                         std::all_of(token.begin(), token.end(),
                                     [](unsigned char ch) { return std::isdigit(ch); });

        if (is_number) {
            int id = std::stoi(token);
            if (id >= 0 && id < static_cast<int>(kCornPuzzleActions.size())) {
                action_id_ = id;
                player_ = Player::kPlayer1;
            }
            return;
        }

        // Console-style action name, e.g. P1R0@0,0.
        if (kCornPuzzleActionNameToID.count(token)) {
            action_id_ = kCornPuzzleActionNameToID[token];
            player_ = Player::kPlayer1;
        }
    }

    inline Player nextPlayer() const override { return Player::kPlayer1; }

    inline std::string toConsoleString() const override
    {
        if (action_id_ < 0 || action_id_ >= static_cast<int>(kCornPuzzleActionName.size())) {
            return "null";
        }
        return kCornPuzzleActionName[action_id_];
    }
};

class CornPuzzleEnv : public BaseEnv<CornPuzzleAction> {
public:
    CornPuzzleEnv() { reset(); }

    void reset() override;
    void resetFromPuzzleFile(const std::string& puzzle_path);
    bool resetFromCurriculumTask(const std::string& puzzle_path,
                                 const std::string& solution_path,
                                 int prefix_piece_count,
                                 const std::string& task_id);
    inline std::string getPuzzlePath() const { return current_puzzle_path_; }
    inline std::string getCurriculumTaskID() const { return curriculum_task_id_; }
    inline std::string getCurriculumSolutionPath() const { return curriculum_solution_path_; }
    inline int getCurriculumPrefix() const { return curriculum_prefix_piece_count_; }

    bool act(const CornPuzzleAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override
    {
        CornPuzzleAction action(action_string_args);
        if (action.getActionID() < 0) {
            return false;
        }
        return act(action);
    }

    std::vector<CornPuzzleAction> getLegalActions() const override;
    bool isLegalAction(const CornPuzzleAction& action) const override;
    bool isTerminal() const override;

    float getReward() const override { return reward_; }
    float getEvalScore(bool is_resign = false) const override
    {
        (void)is_resign;
        return total_reward_;
    }

    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const CornPuzzleAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;

    // Channels:
    //   0: playable empty cells
    //   1: occupied cells
    //   2: blocked cells
    //   3..3+kCornPuzzleNumPieces-1:
    //       candidate geometry planes. Each plane contains the normalized
    //       canonical shape mask for that candidate slot, scaled by
    //       remaining_count / initial_count. This keeps action size fixed
    //       but lets the network see what shape P_i actually represents in
    //       the current puzzle.
    inline int getNumInputChannels() const override { return 3 + kCornPuzzleNumPieces; }

    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return kCornPuzzleRows; }
    inline int getInputChannelWidth() const override { return kCornPuzzleCols; }
    inline int getHiddenChannelHeight() const override { return kCornPuzzleRows; }
    inline int getHiddenChannelWidth() const override { return kCornPuzzleCols; }
    inline int getPolicySize() const override { return static_cast<int>(kCornPuzzleActions.size()); }
    inline int getDiscreteValueSize() const override { return kCornPuzzleDiscreteValueSize; }

    int getRotatePosition(int position, utils::Rotation rotation) const override
    {
        (void)rotation;
        return position;
    }

    int getRotateAction(int action_id, utils::Rotation rotation) const override
    {
        (void)rotation;
        return action_id;
    }

    std::string toString() const override;
    std::string actionToConsoleString(const CornPuzzleAction& action) const;
    std::string name() const override { return kCornPuzzleName; }
    int getNumPlayer() const override { return kCornPuzzleNumPlayer; }

    static void setUpEnv()
    {
        cornpuzzle::initialize();
        config::env_board_size = kCornPuzzleRows;
        config::learner_n_step_return = 10;
        config::zero_actor_intermediate_sequence_length = 200;
    }

private:
    bool canPlace(const CornPlacement& placement) const;
    void place(const CornPlacement& placement);
    bool isSolved() const;
    bool areaMatchesRemaining() const;
    int filledCount() const;

    int activeArea() const;
    void applyCurriculumMask();
    bool applySolutionPrefix(const std::string& solution_path, int prefix_piece_count);
    bool resolvePositionPlacement(const CornPlacement& action_placement,
                                  int position,
                                  CornPlacement& resolved) const;
    bool makeActionPlacement(int action_id, CornPlacement& placement) const;
    bool candidateHasLegalFirstLayerPosition(const CornPlacement& placement) const;
    bool candidateHasForcedRightPlacement(const CornPlacement& placement) const;
    bool makePendingPlacement(CornPlacement& placement) const;

    // Right-trim layer packing state.  The policy/action size stays identical
    // to the original two-stage environment.  Only the first piece of the
    // topmost unfinished row gets a free position choice.  Afterwards the
    // next empty cell encountered while scanning right is the forced anchor.
    bool isLayerRowFull(int row) const;
    int findTopmostUnfinishedRow() const;
    int getForcedLayerTargetPosition() const;
    void initializeLayerStateFromBoard();
    void updateLayerStateAfterPlacement(int placed_position, bool was_first_piece);

private:
    std::array<int, kCornPuzzleBoardSize> board_;
    std::array<int, kCornPuzzleNumPieces> remaining_;
    std::array<int, kCornPuzzleNumPieces> initial_remaining_ = {};
    std::vector<CornCandidateShape> candidate_shapes_;
    int raw_piece_count_ = 0;

    int active_rows_ = kCornPuzzlePlayableRows;
    int active_cols_ = kCornPuzzleCols;
    std::string current_puzzle_path_;
    std::string curriculum_task_id_;
    std::string curriculum_solution_path_;
    int curriculum_prefix_piece_count_ = 0;

    // A complete placement is represented by two consecutive actions while
    // keeping the fixed 128-logit policy head:
    //   SELECT_PIECE:    action = candidate_id * 2 + rotation_id
    //   SELECT_POSITION: action = row * 14 + col (0..97)
    CornActionPhase action_phase_ = CornActionPhase::kSelectPiece;
    int pending_piece_id_ = -1;
    int pending_rotation_ = 0;

    // Layer packing mode (kept outside the policy head so pretrained
    // two-stage weights remain shape-compatible).
    int current_layer_row_ = 0;
    bool first_piece_of_layer_ = true;
    int layer_next_scan_col_ = 0;

    float reward_ = 0.0f;
    float total_reward_ = 0.0f;
};

class CornPuzzleEnvLoader : public BaseEnvLoader<CornPuzzleAction, CornPuzzleEnv> {
public:
    void loadFromEnvironment(const CornPuzzleEnv& env,
                             const std::vector<std::vector<std::pair<std::string, std::string>>>& action_info_history = {}) override
    {
        BaseEnvLoader<CornPuzzleAction, CornPuzzleEnv>::loadFromEnvironment(env, action_info_history);
        addTag("PUZZLE", env.getPuzzlePath());
        if (!env.getCurriculumTaskID().empty()) {
            addTag("CTASK", env.getCurriculumTaskID());
            addTag("CSOLUTION", env.getCurriculumSolutionPath());
            addTag("CPREFIX", std::to_string(env.getCurriculumPrefix()));
        }
    }

    inline std::string getCurriculumTaskID() const { return getTag("CTASK"); }
    inline std::string getCurriculumSolutionPath() const { return getTag("CSOLUTION"); }
    inline int getCurriculumPrefix() const
    {
        const std::string value = getTag("CPREFIX");
        return value.empty() ? 0 : std::stoi(value);
    }

    inline std::string getPuzzlePath() const
    {
        return BaseEnvLoader<CornPuzzleAction, CornPuzzleEnv>::getTag("PUZZLE");
    }

    std::vector<float> getFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;

    std::vector<float> getValue(const int pos) const override
    {
        (void)pos;
        return {getReturn()};
    }

    std::string name() const override { return kCornPuzzleName; }
    int getPolicySize() const override { return static_cast<int>(kCornPuzzleActions.size()); }

    int getRotatePosition(int position, utils::Rotation rotation) const override
    {
        (void)rotation;
        return position;
    }

    int getRotateAction(int action_id, utils::Rotation rotation) const override
    {
        (void)rotation;
        return action_id;
    }
};

} // namespace minizero::env::cornpuzzle
