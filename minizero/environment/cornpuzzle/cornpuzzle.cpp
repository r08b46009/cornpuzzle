#include "cornpuzzle.h"
#include "configuration.h"
#include "random.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace minizero::env::cornpuzzle {

std::vector<CornPlacement> kCornPuzzleActions;
std::vector<std::string> kCornPuzzleActionName;
std::unordered_map<std::string, int> kCornPuzzleActionNameToID;
// export CORNPUZZLE_LIBRARY=/workspace/corn_piece_library.txt
// export CORNPUZZLE_FILE=/workspace/easy_wrap.txt
// export CORNPUZZLE_FILE=/workspace/medium_wrap_4x8.txt
// Runtime-loaded puzzle.
// The puzzle is read from the file pointed to by:
//   export CORNPUZZLE_FILE=/path/to/puzzle.txt
//
// Recommended txt format:
//
//   # rows 4
//   # cols 6
//
//   111
//   001
//
//   11
//
//   1
//   1
//
// Pieces are separated by blank lines.
// Cells can be written as '1' or '#'. Other characters are treated as empty.
// Identical shapes are grouped automatically and become multiple copies of one piece type.
static std::vector<std::vector<std::pair<int, int>>> kBasePieces;
static std::unordered_map<std::string, int> kShapeKeyToPieceID;
static bool gPieceLibraryLoaded = false;
struct CornPuzzleSpec {
    std::array<int, kCornPuzzleNumPieces> initial_remaining = {};
    std::vector<CornCandidateShape> candidate_shapes;
    int active_rows = kCornPuzzlePlayableRows;
    int active_cols = kCornPuzzleCols;
    int total_area = 0;
    int raw_piece_count = 0;
};


static std::string getConfiguredPuzzleDir()
{
    return minizero::config::env_compound_puzzles_dir;
}

static bool useCompoundPuzzleFolder()
{
    return !getConfiguredPuzzleDir().empty();
}

static std::string selectPuzzleFileFromFolder(const std::string& folder_name)
{
    std::vector<std::string> files;

    for (const auto& entry : std::filesystem::directory_iterator(folder_name)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }

    if (files.empty()) {
        std::cerr << "[CornPuzzle] No puzzle files found in directory: "
                  << folder_name << std::endl;
        std::exit(1);
    }

    // Keep deterministic ordering when random selection is disabled.
    std::sort(files.begin(), files.end());

    if (minizero::config::env_compound_random_select_puzzle) {
        int idx = minizero::utils::Random::randInt() % static_cast<int>(files.size());
        return files[idx];
    }

    return files[0];
}

static std::string getRequiredLibraryPath()
{
    if (minizero::config::env_compound_piece_library.empty()) {
        std::cerr << "[CornPuzzle] Please set env_compound_piece_library in cfg."
                  << std::endl;
        std::exit(1);
    }

    return minizero::config::env_compound_piece_library;
}

// ----------------------------
// Shape utilities
// ----------------------------

static std::vector<std::pair<int, int>> normalize(std::vector<std::pair<int, int>> cells)
{
    if (cells.empty()) {
        return cells;
    }

    int min_r = 999;
    int min_c = 999;

    for (auto [r, c] : cells) {
        min_r = std::min(min_r, r);
        min_c = std::min(min_c, c);
    }

    for (auto& [r, c] : cells) {
        r -= min_r;
        c -= min_c;
    }

    std::sort(cells.begin(), cells.end());
    return cells;
}

static std::vector<std::pair<int, int>> rotate90(const std::vector<std::pair<int, int>>& cells)
{
    int max_r = 0;

    for (auto [r, c] : cells) {
        max_r = std::max(max_r, r);
    }

    std::vector<std::pair<int, int>> out;

    for (auto [r, c] : cells) {
        out.push_back({c, max_r - r});
    }

    return normalize(out);
}

static std::vector<std::pair<int, int>> applyRotation(std::vector<std::pair<int, int>> cells, int rot)
{
    cells = normalize(cells);

    int times = rot / 90;
    for (int i = 0; i < times; ++i) {
        cells = rotate90(cells);
    }

    return normalize(cells);
}
static std::string shapeKey(const std::vector<std::pair<int, int>>& cells)
{
    std::ostringstream oss;
    auto norm = normalize(cells);

    for (auto [r, c] : norm) {
        oss << r << "," << c << ";";
    }

    return oss.str();
}
static std::string canonicalShapeKey180(const std::vector<std::pair<int, int>>& cells)
{
    std::string key0 = shapeKey(cells);

    auto rot180 = applyRotation(cells, 180);
    std::string key180 = shapeKey(rot180);

    return std::min(key0, key180);
}

static std::vector<std::pair<int, int>> canonicalCells180(const std::vector<std::pair<int, int>>& cells)
{
    auto norm = normalize(cells);
    auto rot180 = applyRotation(cells, 180);

    if (shapeKey(rot180) < shapeKey(norm)) {
        return rot180;
    }

    return norm;
}

static void computeShapeBounds(CornCandidateShape& shape)
{
    shape.area = static_cast<int>(shape.cells.size());
    shape.height = 0;
    shape.width = 0;

    for (auto [r, c] : shape.cells) {
        shape.height = std::max(shape.height, r + 1);
        shape.width = std::max(shape.width, c + 1);
    }
}
static int wrapCol(int c)
{
    c %= kCornPuzzleCols;

    if (c < 0) {
        c += kCornPuzzleCols;
    }

    return c;
}

static int wrapColActive(int c, int active_cols)
{
    c %= active_cols;

    if (c < 0) {
        c += active_cols;
    }

    return c;
}

static bool inBoundsFullRowsOnly(const std::vector<std::pair<int, int>>& cells, int top)
{
    for (auto [dr, dc] : cells) {
        (void)dc;

        int r = top + dr;

        // Row does not wrap.
        if (r < 0 || r >= kCornPuzzlePlayableRows) {
            return false;
        }
    }

    // Columns wrap, so full-board left/right bounds are not checked here.
    return true;
}

static std::vector<std::pair<int, int>> pieceFromGrid(const std::vector<std::string>& lines)
{
    std::vector<std::pair<int, int>> cells;

    for (int r = 0; r < static_cast<int>(lines.size()); ++r) {
        for (int c = 0; c < static_cast<int>(lines[r].size()); ++c) {
            if (lines[r][c] == '1' || lines[r][c] == '#') {
                cells.push_back({r, c});
            }
        }
    }

    return normalize(cells);
}

static std::string cornPuzzleBoardLogPath()
{
    if (!minizero::config::zero_training_directory.empty()) {
        return minizero::config::zero_training_directory + "/cornpuzzle_board_debug.log";
    }

    return "cornpuzzle_board_debug.log";
}

static void updateActionCellsFromLibrary()
{
    for (auto& action : kCornPuzzleActions) {
        int p = action.piece_id;

        if (p < 0 || p >= static_cast<int>(kBasePieces.size())) {
            action.cells.clear();
            continue;
        }

        action.cells = applyRotation(kBasePieces[p], action.rotation);
    }
}

// ----------------------------
// Puzzle txt loading
// ----------------------------
static void loadPieceLibraryFromTxt(const std::string& path)
{
    kBasePieces.clear();
    kShapeKeyToPieceID.clear();

    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "[CornPuzzle] Failed to open piece library file: "
                  << path << std::endl;
        std::exit(1);
    }

    std::vector<std::string> current;
    std::string line;

    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty() && line[0] == '#') {
            continue;
        }

        if (line.empty()) {
            if (!current.empty()) {
                auto cells = pieceFromGrid(current);
                if (!cells.empty()) {
                    auto norm = normalize(cells);
                    std::string key = canonicalShapeKey180(norm);

                    if (!kShapeKeyToPieceID.count(key)) {
                        int id = static_cast<int>(kBasePieces.size());
                        kBasePieces.push_back(norm);
                        kShapeKeyToPieceID[key] = id;
                    }
                }
                current.clear();
            }
            continue;
        }

        current.push_back(line);
    }

    if (!current.empty()) {
        auto cells = pieceFromGrid(current);
        if (!cells.empty()) {
            auto norm = normalize(cells);
            std::string key = canonicalShapeKey180(norm);

            if (!kShapeKeyToPieceID.count(key)) {
                int id = static_cast<int>(kBasePieces.size());
                kBasePieces.push_back(norm);
                kShapeKeyToPieceID[key] = id;
            }
        }
    }

    if (static_cast<int>(kBasePieces.size()) > kCornPuzzleNumPieces) {
        std::cerr << "[CornPuzzle] Too many piece types in library: "
                  << kBasePieces.size()
                  << ". Max supported = "
                  << kCornPuzzleNumPieces
                  << std::endl;
        std::exit(1);
    }

    while (static_cast<int>(kBasePieces.size()) < kCornPuzzleNumPieces) {
        kBasePieces.push_back({{0, 0}});
    }

    gPieceLibraryLoaded = true;

    updateActionCellsFromLibrary();

    std::cerr << "[CornPuzzle] Loaded fixed piece library from "
              << path
              << " | piece_types=" << kBasePieces.size()
              << std::endl;
}
static CornPuzzleSpec loadPuzzleSpecFromTxt(const std::string& path)
{
    CornPuzzleSpec spec;
    spec.initial_remaining.fill(0);
    spec.active_rows = kCornPuzzlePlayableRows;
    spec.active_cols = kCornPuzzleCols;

    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "[CornPuzzle] Failed to open puzzle file: " << path << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<std::pair<int, int>>> raw_pieces;
    std::vector<std::string> current;
    std::string line;

    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Header examples:
        //   # rows 4
        //   # cols 6
        //   # active_rows 4
        //   # active_cols 6
        if (!line.empty() && line[0] == '#') {
            std::istringstream iss(line.substr(1));
            std::string key;
            int value = 0;
            iss >> key >> value;

            if (key == "rows" || key == "active_rows") {
                spec.active_rows = value;
            } else if (key == "cols" || key == "active_cols") {
                spec.active_cols = value;
            }

            continue;
        }

        if (line.empty()) {
            if (!current.empty()) {
                auto cells = pieceFromGrid(current);
                if (!cells.empty()) {
                    raw_pieces.push_back(cells);
                }
                current.clear();
            }
            continue;
        }

        current.push_back(line);
    }

    if (!current.empty()) {
        auto cells = pieceFromGrid(current);
        if (!cells.empty()) {
            raw_pieces.push_back(cells);
        }
    }

    if (spec.active_rows <= 0 || spec.active_rows > kCornPuzzlePlayableRows) {
        std::cerr << "[CornPuzzle] Invalid active rows: " << spec.active_rows
                  << ". It must be between 1 and " << kCornPuzzlePlayableRows << std::endl;
        std::exit(1);
    }

    if (spec.active_cols <= 0 || spec.active_cols > kCornPuzzleCols) {
        std::cerr << "[CornPuzzle] Invalid active cols: " << spec.active_cols
                  << ". It must be between 1 and " << kCornPuzzleCols << std::endl;
        std::exit(1);
    }

    if (raw_pieces.empty()) {
        std::cerr << "[CornPuzzle] No pieces found in puzzle file: " << path << std::endl;
        std::exit(1);
    }

    // Candidate-action import layer:
    // each puzzle normalizes, canonicalizes, and groups its own pieces.
    // The sorted canonical shape key gives a deterministic candidate_index
    // within the puzzle. The network can still see the selected action shape
    // through getActionFeatures(), so the action index is no longer intended
    // to carry fixed global geometry semantics by itself.
    struct GroupedShape {
        std::vector<std::pair<int, int>> cells;
        int count = 0;
    };

    std::map<std::string, GroupedShape> grouped;
    for (const auto& cells : raw_pieces) {
        auto canonical_cells = canonicalCells180(cells);
        std::string key = shapeKey(canonical_cells);

        auto& group = grouped[key];
        if (group.cells.empty()) {
            group.cells = canonical_cells;
        }
        group.count++;
    }

    if (static_cast<int>(grouped.size()) > kCornPuzzleNumPieces) {
        std::cerr << "[CornPuzzle] Too many unique candidate shapes in puzzle: "
                  << grouped.size()
                  << ". Max supported = " << kCornPuzzleNumPieces
                  << std::endl;
        std::exit(1);
    }

    int candidate_id = 0;
    spec.total_area = 0;
    for (const auto& item : grouped) {
        CornCandidateShape shape;
        shape.candidate_id = candidate_id;
        shape.count = item.second.count;
        shape.shape_key = item.first;
        shape.cells = normalize(item.second.cells);
        computeShapeBounds(shape);

        spec.candidate_shapes.push_back(shape);
        spec.initial_remaining[candidate_id] = shape.count;
        spec.total_area += shape.count * shape.area;
        candidate_id++;
    }

    spec.raw_piece_count = static_cast<int>(raw_pieces.size());

    int active_area = spec.active_rows * spec.active_cols;
    if (spec.total_area != active_area) {
        std::cerr << "[CornPuzzle] Warning: puzzle " << path
                  << " piece area = " << spec.total_area
                  << ", active board area = " << active_area
                  << ". This puzzle may terminate immediately by area mismatch."
                  << std::endl;
    }

    return spec;
}

// ----------------------------
// Active-wrap legality helpers
// ----------------------------

static bool canPlaceOnBoard(
    const std::array<int, kCornPuzzleBoardSize>& board,
    const std::array<int, kCornPuzzleNumPieces>& remaining,
    const CornPlacement& placement,
    int active_rows,
    int active_cols)
{
    if (placement.piece_id < 0 || placement.piece_id >= kCornPuzzleNumPieces) {
        return false;
    }


    if (remaining[placement.piece_id] <= 0) {
        return false;
    }

    // Avoid duplicated equivalent actions when active_cols < 14.
    if (placement.top < 0 || placement.top >= active_rows) {
        return false;
    }

    if (placement.left < 0 || placement.left >= active_cols) {
        return false;
    }

    std::set<int> occupied_positions;

    for (auto [dr, dc] : placement.cells) {
        int r = placement.top + dr;
        int c = wrapColActive(placement.left + dc, active_cols);

        if (r < 0 || r >= active_rows) {
            return false;
        }

        int pos = r * kCornPuzzleCols + c;

        // Prevent self-overlap after active column wrapping.
        if (occupied_positions.count(pos)) {
            return false;
        }

        occupied_positions.insert(pos);

        // board[pos] == -1 means blocked.
        // board[pos] > 0 means occupied.
        // Only 0 is placeable.
        if (board[pos] != 0) {
            return false;
        }
    }

    return true;
}

// ----------------------------
// Environment initialization
// ----------------------------

void initialize()
{
    kCornPuzzleActions.clear();
    kCornPuzzleActionName.clear();
    kCornPuzzleActionNameToID.clear();

    // Sokoban-style initialization:
    // Create a fixed action space only. Do NOT load library or puzzle here,
    // because MiniZero may call initialize() before the cfg values are loaded.
    for (int p = 0; p < kCornPuzzleNumPieces; ++p) {
        for (int rot_idx = 0; rot_idx < kCornPuzzleNumRotations; ++rot_idx) {
            int rot = (rot_idx == 0) ? 0 : 180;
            CornPlacement placement;
            placement.piece_id = p;
            placement.rotation = rot;
            placement.top = 0;
            placement.left = 0;
            placement.cells.clear();  // filled after the library is loaded in reset()

            int action_id = static_cast<int>(kCornPuzzleActions.size());

            std::ostringstream name;
            name << "P" << (p + 1) << "R" << rot;

            kCornPuzzleActions.push_back(placement);
            kCornPuzzleActionName.push_back(name.str());
            kCornPuzzleActionNameToID[name.str()] = action_id;
        }
    }

    if (gPieceLibraryLoaded) {
        updateActionCellsFromLibrary();
    }

    kCornPuzzleActionName.push_back("null");

    std::cerr << "[CornPuzzle] Initialized fixed actions: "
              << kCornPuzzleActions.size()
              << " + null"
              << std::endl;
}

// ----------------------------
// CornPuzzleEnv
// ----------------------------

void CornPuzzleEnv::reset()
{
    std::string puzzle_dir = getConfiguredPuzzleDir();
    if (puzzle_dir.empty()) {
        std::cerr << "[CornPuzzle] Please set env_compound_puzzles_dir in cfg."
                  << std::endl;
        std::exit(1);
    }

    std::string puzzle_path = selectPuzzleFileFromFolder(puzzle_dir);
    resetFromPuzzleFile(puzzle_path);
}

void CornPuzzleEnv::resetFromPuzzleFile(const std::string& puzzle_path)
{
    CornPuzzleSpec spec = loadPuzzleSpecFromTxt(puzzle_path);

    current_puzzle_path_ = puzzle_path;

    turn_ = Player::kPlayer1;
    actions_.clear();
    observations_.clear();

    active_rows_ = spec.active_rows;
    active_cols_ = spec.active_cols;

    initial_remaining_ = spec.initial_remaining;
    remaining_ = spec.initial_remaining;
    candidate_shapes_ = spec.candidate_shapes;

    applyCurriculumMask();

    reward_ = 0.0f;
    total_reward_ = 0.0f;
}

void CornPuzzleEnv::applyCurriculumMask()
{
    board_.fill(-1);

    for (int r = 0; r < active_rows_; ++r) {
        for (int c = 0; c < active_cols_; ++c) {
            board_[r * kCornPuzzleCols + c] = 0;
        }
    }
}

int CornPuzzleEnv::activeArea() const
{
    return active_rows_ * active_cols_;
}

bool CornPuzzleEnv::canPlace(const CornPlacement& placement) const
{
    return canPlaceOnBoard(board_, remaining_, placement, active_rows_, active_cols_);
}

void CornPuzzleEnv::place(const CornPlacement& placement)
{
    int mark = placement.piece_id + 1;

    for (auto [dr, dc] : placement.cells) {
        int r = placement.top + dr;
        int c = wrapColActive(placement.left + dc, active_cols_);
        board_[r * kCornPuzzleCols + c] = mark;
    }

    remaining_[placement.piece_id]--;
}

int CornPuzzleEnv::filledCount() const
{
    int filled = 0;

    for (int r = 0; r < active_rows_; ++r) {
        for (int c = 0; c < active_cols_; ++c) {
            if (board_[r * kCornPuzzleCols + c] > 0) {
                filled++;
            }
        }
    }

    return filled;
}

int CornPuzzleEnv::firstEmptyPos() const
{
    for (int r = 0; r < active_rows_; ++r) {
        for (int c = 0; c < active_cols_; ++c) {
            int pos = r * kCornPuzzleCols + c;
            if (board_[pos] == 0) {
                return pos;
            }
        }
    }

    return -1;
}

bool CornPuzzleEnv::resolveFirstEmptyPlacement(
    const CornPlacement& action_placement,
    CornPlacement& resolved) const
{
    int target_pos = firstEmptyPos();
    if (target_pos < 0) {
        return false;
    }

    if (action_placement.cells.empty()) {
        return false;
    }

    int target_r = target_pos / kCornPuzzleCols;
    int target_c = target_pos % kCornPuzzleCols;

    resolved = action_placement;

    // Deterministic placement rule:
    // align the first normalized cell of the selected rotated piece to the
    // current first-empty board cell. Columns still use active wrapping.
    auto [anchor_dr, anchor_dc] = resolved.cells[0];

    resolved.top = target_r - anchor_dr;
    resolved.left = wrapColActive(target_c - anchor_dc, active_cols_);

    return true;
}

bool CornPuzzleEnv::makeActionPlacement(int action_id, CornPlacement& placement) const
{
    if (action_id < 0 || action_id >= static_cast<int>(kCornPuzzleActions.size())) {
        return false;
    }

    int candidate_id = action_id / kCornPuzzleNumRotations;
    int rot_idx = action_id % kCornPuzzleNumRotations;

    if (candidate_id < 0 || candidate_id >= static_cast<int>(candidate_shapes_.size())) {
        return false;
    }

    if (candidate_id >= kCornPuzzleNumPieces) {
        return false;
    }

    int rot = (rot_idx == 0) ? 0 : 180;

    placement.piece_id = candidate_id;
    placement.rotation = rot;
    placement.top = 0;
    placement.left = 0;
    placement.cells = applyRotation(candidate_shapes_[candidate_id].cells, rot);

    return !placement.cells.empty();
}

bool CornPuzzleEnv::isSolved() const
{
    if (filledCount() != activeArea()) {
        return false;
    }

    for (int rem : remaining_) {
        if (rem != 0) {
            return false;
        }
    }

    return true;
}

bool CornPuzzleEnv::areaMatchesRemaining() const
{
    int empty = activeArea() - filledCount();

    int need = 0;
    for (int p = 0; p < static_cast<int>(candidate_shapes_.size()); ++p) {
        need += remaining_[p] * candidate_shapes_[p].area;
    }

    return empty == need;
}

bool CornPuzzleEnv::act(const CornPuzzleAction& action)
{
    if (!isLegalAction(action)) {
        return false;
    }

    CornPlacement action_placement;
    if (!makeActionPlacement(action.getActionID(), action_placement)) {
        return false;
    }

    CornPlacement resolved;
    if (!resolveFirstEmptyPlacement(action_placement, resolved)) {
        return false;
    }

    int filled_before = filledCount();

    place(resolved);
    actions_.push_back(action);

    int filled_after = filledCount();

    float progress_before = static_cast<float>(filled_before) / static_cast<float>(std::max(1, activeArea()));
    float progress_after  = static_cast<float>(filled_after)  / static_cast<float>(std::max(1, activeArea()));
    float delta_progress  = progress_after - progress_before;

    // Keep these disabled for the txt + active-wrap version.
    // The older local heuristic helpers used full-board 14-column wrap.
    bool local_zero_candidate = false;
    float mobility_penalty = 0.0f;

    // Bounded reward design:
    // Each placed area contributes at most 0.5 total progress reward.
    // Solved board gets an additional +0.5.
    // Therefore total_reward_ should roughly stay within 0~1.
    reward_ = 0.5f * delta_progress;

    bool solved = isSolved();
    bool area_mismatch = !areaMatchesRemaining();
    bool no_legal_actions = getLegalActions().empty();
    bool dead_end = (!solved && (area_mismatch || no_legal_actions));

    if (solved) {
        reward_ += 0.5f;
    }

    total_reward_ += reward_;

    static int terminal_print_count = 0;
    static int terminal_board_count = 0;

    if (solved || dead_end) {
        terminal_board_count++;

        int legal_action_count = getLegalActions().size();

        if (terminal_print_count < 50) {
            std::cerr << "[CornPuzzle terminal]"
                      << " terminal_index=" << terminal_board_count
                      << ", solved=" << solved
                      << ", dead_end=" << dead_end
                      << ", area_mismatch=" << area_mismatch
                      << ", no_legal_actions=" << no_legal_actions
                      << ", local_zero_candidate=" << local_zero_candidate
                      << ", mobility_penalty=" << mobility_penalty
                      << ", active_rows=" << active_rows_
                      << ", active_cols=" << active_cols_
                      << ", filled=" << filled_after
                      << ", remaining_empty=" << (activeArea() - filled_after)
                      << ", last_action_id=" << action.getActionID()
                      << ", last_action=" << action.toConsoleString()
                      << ", reward=" << reward_
                      << ", total_reward=" << total_reward_
                      << ", legal_actions=" << legal_action_count
                      << std::endl;

            terminal_print_count++;
        }

        if (terminal_board_count % 10000 == 0) {
            std::ofstream debug_log(cornPuzzleBoardLogPath(), std::ios::app);

            debug_log << "\n[CornPuzzle terminal board]"
                      << " terminal_index=" << terminal_board_count
                      << ", solved=" << solved
                      << ", dead_end=" << dead_end
                      << ", area_mismatch=" << area_mismatch
                      << ", no_legal_actions=" << no_legal_actions
                      << ", local_zero_candidate=" << local_zero_candidate
                      << ", mobility_penalty=" << mobility_penalty
                      << ", active_rows=" << active_rows_
                      << ", active_cols=" << active_cols_
                      << ", filled=" << filled_after
                      << ", remaining_empty=" << (activeArea() - filled_after)
                      << ", last_action_id=" << action.getActionID()
                      << ", last_action=" << action.toConsoleString()
                      << ", reward=" << reward_
                      << ", total_reward=" << total_reward_
                      << ", legal_actions=" << legal_action_count
                      << "\n";

            debug_log << toString() << "\n";
            debug_log.flush();
        }
    }

    turn_ = Player::kPlayer1;
    return true;
}

std::vector<CornPuzzleAction> CornPuzzleEnv::getLegalActions() const
{
    std::vector<CornPuzzleAction> actions;

    if (turn_ != Player::kPlayer1) {
        return actions;
    }

    // Reduced action-space version:
    // Each action only selects piece type + rotation. The position is resolved
    // by aligning the piece to the current first-empty cell.
    for (int action_id = 0; action_id < static_cast<int>(kCornPuzzleActions.size()); ++action_id) {
        CornPuzzleAction action(action_id, Player::kPlayer1);

        if (isLegalAction(action)) {
            actions.push_back(action);
        }
    }

    return actions;
}

bool CornPuzzleEnv::isLegalAction(const CornPuzzleAction& action) const
{
    int action_id = action.getActionID();

    if (action.getPlayer() != Player::kPlayer1) {
        return false;
    }

    if (action_id < 0 || action_id >= static_cast<int>(kCornPuzzleActions.size())) {
        return false;
    }

    CornPlacement action_placement;
    if (!makeActionPlacement(action_id, action_placement)) {
        return false;
    }

    if (action_placement.cells.empty()) {
        return false;
    }

    CornPlacement resolved;
    if (!resolveFirstEmptyPlacement(action_placement, resolved)) {
        return false;
    }

    return canPlace(resolved);
}

bool CornPuzzleEnv::isTerminal() const
{
    if (isSolved()) {
        return true;
    }

    if (!areaMatchesRemaining()) {
        return true;
    }

    return getLegalActions().empty();
}

std::vector<float> CornPuzzleEnv::getFeatures(utils::Rotation rotation) const
{
    (void)rotation;

    std::vector<float> features;
    features.reserve(getNumInputChannels() * kCornPuzzleBoardSize);

    // Channel 0: playable empty cells
    for (int pos = 0; pos < kCornPuzzleBoardSize; ++pos) {
        features.push_back(board_[pos] == 0 ? 1.0f : 0.0f);
    }

    // Channel 1: occupied cells
    for (int pos = 0; pos < kCornPuzzleBoardSize; ++pos) {
        features.push_back(board_[pos] > 0 ? 1.0f : 0.0f);
    }

    // Channel 2: blocked cells
    for (int pos = 0; pos < kCornPuzzleBoardSize; ++pos) {
        features.push_back(board_[pos] < 0 ? 1.0f : 0.0f);
    }

    // Channels 3..: candidate geometry planes, not just count planes.
    //
    // Each candidate slot P_i can mean a different normalized shape in each
    // puzzle. Therefore the slot ID alone is not meaningful. To make the
    // fixed action space learnable, we expose the actual canonical geometry
    // of candidate i in channel 3+i. The mask is placed at the origin of the
    // feature board and scaled by normalized remaining count, so:
    //   value = min(remaining_count / kCornPuzzleRemainingCountNorm, 1.0)
    //   - zero plane       => candidate absent or exhausted
    //   - nonzero mask     => candidate shape and absolute remaining availability
    //
    // Rotation is intentionally NOT integrated here. Rotation-specific
    // geometry is still available to MuZero through getActionFeatures(action)
    // during recurrent unrolling. This keeps the state feature compact and
    // follows the original candidate_index x rotation action design.
    for (int p = 0; p < kCornPuzzleNumPieces; ++p) {
        std::array<float, kCornPuzzleBoardSize> plane = {};
        plane.fill(0.0f);

        if (p < static_cast<int>(candidate_shapes_.size()) && remaining_[p] > 0) {
            float denom = static_cast<float>(std::max(1, kCornPuzzleRemainingCountNorm));
            float v = static_cast<float>(remaining_[p]) / denom;
            v = std::min(v, 1.0f);

            for (auto [dr, dc] : candidate_shapes_[p].cells) {
                if (dr >= 0 && dr < kCornPuzzleRows && dc >= 0 && dc < kCornPuzzleCols) {
                    plane[dr * kCornPuzzleCols + dc] = v;
                }
            }
        }

        for (int pos = 0; pos < kCornPuzzleBoardSize; ++pos) {
            features.push_back(plane[pos]);
        }
    }

    return features;
}

std::vector<float> CornPuzzleEnv::getActionFeatures(const CornPuzzleAction& action, utils::Rotation rotation) const
{
    (void)rotation;

    std::vector<float> action_features(kCornPuzzleBoardSize, 0.0f);

    if (action.getActionID() < 0 || action.getActionID() >= static_cast<int>(kCornPuzzleActions.size())) {
        return action_features;
    }

    CornPlacement placement;
    if (!makeActionPlacement(action.getActionID(), placement)) {
        return action_features;
    }

    // Because the reduced action no longer contains an explicit board position,
    // the action feature represents only the selected rotated candidate shape.
    // The environment will decide the real position from the current first-empty
    // cell when act() / isLegalAction() is called.
    for (auto [dr, dc] : placement.cells) {
        if (dr >= 0 && dr < kCornPuzzleRows && dc >= 0 && dc < kCornPuzzleCols) {
            action_features[dr * kCornPuzzleCols + dc] = 1.0f;
        }
    }

    return action_features;
}


static bool useCornPuzzleColorOutput()
{
    const char* flag = std::getenv("CORNPUZZLE_COLOR");

    if (flag == nullptr) {
        return false;
    }

    std::string value(flag);
    return value != "0" && value != "false" && value != "FALSE";
}

static const char* cornPieceAnsiColor(int piece_value)
{
    // piece_value is board mark = piece_id + 1.
    // Use bright ANSI colors and cycle if there are more piece types.
    static const std::array<const char*, 16> colors = {
        "\033[38;5;196m", // red
        "\033[38;5;208m", // orange
        "\033[38;5;226m", // yellow
        "\033[38;5;46m",  // green
        "\033[38;5;51m",  // cyan
        "\033[38;5;33m",  // blue
        "\033[38;5;129m", // purple
        "\033[38;5;201m", // magenta
        "\033[38;5;118m", // light green
        "\033[38;5;45m",  // turquoise
        "\033[38;5;214m", // gold
        "\033[38;5;141m", // violet
        "\033[38;5;160m", // dark red
        "\033[38;5;39m",  // sky blue
        "\033[38;5;220m", // amber
        "\033[38;5;82m"   // neon green
    };

    if (piece_value <= 0) {
        return "";
    }

    return colors[(piece_value - 1) % colors.size()];
}

static std::string cornColorizeToken(const std::string& token, int board_value, bool use_color)
{
    if (!use_color) {
        return token;
    }

    constexpr const char* reset = "\033[0m";
    constexpr const char* blocked = "\033[38;5;240m";

    if (board_value < 0) {
        return std::string(blocked) + token + reset;
    }

    if (board_value == 0) {
        return token;
    }

    return std::string(cornPieceAnsiColor(board_value)) + token + reset;
}

static std::string cornCellToken(int board_value)
{
    std::ostringstream cell;

    if (board_value < 0) {
        cell << " X";
    } else if (board_value == 0) {
        cell << " .";
    } else {
        cell << std::setw(2) << board_value;
    }

    return cell.str();
}

std::string CornPuzzleEnv::toString() const
{
    const bool use_color = useCornPuzzleColorOutput();

    std::ostringstream oss;

    oss << "active_rows=" << active_rows_
        << ", active_cols=" << active_cols_
        << "\n";

    oss << "    ";
    for (int c = 0; c < kCornPuzzleCols; ++c) {
        oss << std::setw(2) << c << " ";
    }
    oss << "\n";

    for (int r = 0; r < kCornPuzzlePlayableRows; ++r) {
        oss << std::setw(2) << r << "  ";

        for (int c = 0; c < kCornPuzzleCols; ++c) {
            int v = board_[r * kCornPuzzleCols + c];
            std::string token = cornCellToken(v);
            oss << cornColorizeToken(token, v, use_color) << " ";
        }

        oss << "\n";
    }

    oss << "remaining:";
    for (int p = 0; p < kCornPuzzleNumPieces; ++p) {
        int piece_value = p + 1;

        std::ostringstream label;
        label << "P" << piece_value;

        if (use_color) {
            oss << " "
                << cornColorizeToken(label.str(), piece_value, true)
                << "=" << remaining_[p];
        } else {
            oss << " " << label.str() << "=" << remaining_[p];
        }
    }
    oss << "\n";

    return oss.str();
}

std::vector<float> CornPuzzleEnvLoader::getFeatures(const int pos, utils::Rotation rotation) const
{
    CornPuzzleEnv env;

    if (!getPuzzlePath().empty()) {
        env.resetFromPuzzleFile(getPuzzlePath());
    }

    for (int i = 0; i < std::min(pos, static_cast<int>(action_pairs_.size())); ++i) {
        env.act(action_pairs_[i].first);
    }

    return env.getFeatures(rotation);
}

std::vector<float> CornPuzzleEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation) const
{
    CornPuzzleEnv env;

    if (!getPuzzlePath().empty()) {
        env.resetFromPuzzleFile(getPuzzlePath());
    }

    for (int i = 0; i < std::min(pos, static_cast<int>(action_pairs_.size())); ++i) {
        env.act(action_pairs_[i].first);
    }

    return env.getActionFeatures(
        pos < static_cast<int>(action_pairs_.size()) ? action_pairs_[pos].first : CornPuzzleAction(),
        rotation
    );
}

} // namespace minizero::env::cornpuzzle
