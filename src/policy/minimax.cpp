#include <algorithm>
#include <vector>
#include <utility>
#include "state.hpp"
#include "minimax.hpp"

static const int move_order_piece_value[7] = {0, 20, 60, 70, 80, 200, 10000};

static int move_order_score(const State *state, const Move &move)
{
    int self = state->player;
    int opp = 1 - self;
    int from_r = static_cast<int>(move.first.first);
    int from_c = static_cast<int>(move.first.second);
    int to_r = static_cast<int>(move.second.first);
    int to_c = static_cast<int>(move.second.second);
    int moved = state->piece_at(self, from_r, from_c);
    int captured = state->piece_at(opp, to_r, to_c);
    int score = 10 * move_order_piece_value[captured] - move_order_piece_value[moved];

    if (moved == 1 && (to_r == 0 || to_r == state->board_h() - 1))
    {
        score += move_order_piece_value[5];
    }

    return score;
}

static bool captures_king(const State *state, const Move &move)
{
    int opp = 1 - state->player;
    int to_r = static_cast<int>(move.second.first);
    int to_c = static_cast<int>(move.second.second);
    return state->piece_at(opp, to_r, to_c) == 6;
}

static bool legal_move_exists(const State *state, const Move &move)
{
    return std::find(state->legal_actions.begin(), state->legal_actions.end(), move)
        != state->legal_actions.end();
}

static bool try_opening_book(State *state, SearchResult &result)
{
    if (state->player != 1)
    {
        return false;
    }

    Move book_move;
    bool found = false;

    // Against the common b2-b3 opening, occupying c4 gives black a stable
    // central pawn and avoids the weaker d5-d4 line seen in timed games.
    if (state->piece_at(0, 3, 1) == 1
        && state->piece_at(1, 1, 2) == 1
        && state->piece_at(0, 2, 2) == 0
        && state->piece_at(1, 2, 2) == 0)
    {
        book_move = Move(Point(1, 2), Point(2, 2)); // c5c4
        found = true;
    }
    else if (state->piece_at(0, 2, 2) == 1
        && state->piece_at(1, 1, 1) == 1)
    {
        book_move = Move(Point(1, 1), Point(2, 2)); // b5c4
        found = true;
    }
    else if (state->piece_at(0, 3, 2) == 1
        && state->piece_at(1, 1, 0) == 1
        && state->piece_at(0, 2, 0) == 0
        && state->piece_at(1, 2, 0) == 0)
    {
        book_move = Move(Point(1, 0), Point(2, 0)); // a5a4
        found = true;
    }

    if (found && legal_move_exists(state, book_move))
    {
        result.best_move = book_move;
        result.score = 0;
        result.seldepth = 1;
        result.nodes = 1;
        result.pv = {result.best_move};
        return true;
    }
    return false;
}

static bool is_capture_or_promotion(const State *state, const Move &move)
{
    int self = state->player;
    int opp = 1 - self;
    int from_r = static_cast<int>(move.first.first);
    int from_c = static_cast<int>(move.first.second);
    int to_r = static_cast<int>(move.second.first);
    int to_c = static_cast<int>(move.second.second);
    int moved = state->piece_at(self, from_r, from_c);
    return state->piece_at(opp, to_r, to_c) != 0
        || (moved == 1 && (to_r == 0 || to_r == state->board_h() - 1));
}

static std::vector<Move> ordered_actions(const State *state, bool tactical_only)
{
    std::vector<Move> actions;
    actions.reserve(state->legal_actions.size());
    for (const auto &action : state->legal_actions)
    {
        if (!tactical_only || is_capture_or_promotion(state, action))
        {
            actions.push_back(action);
        }
    }

    std::sort(actions.begin(), actions.end(), [state](const Move &a, const Move &b) {
        return move_order_score(state, a) > move_order_score(state, b);
    });
    return actions;
}

static std::vector<Move> filter_suicidal_actions(State *state, const std::vector<Move> &actions)
{
    std::vector<Move> safe_actions;
    safe_actions.reserve(actions.size());

    for (const auto &action : actions)
    {
        State *next = state->next_state(action);
        next->get_legal_actions();
        if (next->game_state != WIN)
        {
            safe_actions.push_back(action);
        }
        delete next;
    }

    return safe_actions.empty() ? actions : safe_actions;
}

static int promotion_distance(const State *state, int player, int row)
{
    return player == 0 ? row : state->board_h() - 1 - row;
}

static bool is_passed_pawn(const State *state, int player, int row, int col)
{
    int opp = 1 - player;
    int step = player == 0 ? -1 : 1;

    for (int r = row + step; r >= 0 && r < state->board_h(); r += step)
    {
        for (int dc = -1; dc <= 1; dc++)
        {
            int c = col + dc;
            if (c >= 0 && c < state->board_w() && state->piece_at(opp, r, c) == 1)
            {
                return false;
            }
        }
    }
    return true;
}

static int pawn_race_score(const State *state, int owner)
{
    int score = 0;
    for (int r = 0; r < state->board_h(); r++)
    {
        for (int c = 0; c < state->board_w(); c++)
        {
            if (state->piece_at(owner, r, c) != 1)
            {
                continue;
            }

            int dist = promotion_distance(state, owner, r);
            int advance = state->board_h() - 1 - dist;
            int bonus = advance * advance * 3;

            if (is_passed_pawn(state, owner, r, c))
            {
                bonus += 12 + advance * 8;
            }
            if (dist <= 2)
            {
                bonus += (3 - dist) * 45;
            }
            if (dist == 0)
            {
                bonus += 160;
            }
            score += bonus;
        }
    }
    return score;
}

static int policy_evaluate(State *state, const MMParams &p, const GameHistory *history)
{
    int score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, history);
    if (!p.use_policy_endgame)
    {
        return score;
    }

    int self = state->player;
    int opp = 1 - self;
    score += pawn_race_score(state, self);
    score -= pawn_race_score(state, opp);
    return score;
}

static int quiescence(
    State *state,
    int alpha,
    int beta,
    GameHistory &history,
    int ply,
    SearchContext &ctx,
    const MMParams &p)
{
    ctx.nodes++;
    if (ply > ctx.seldepth)
    {
        ctx.seldepth = ply;
    }
    if (ctx.stop)
    {
        return 0;
    }

    if (state->legal_actions.empty() && state->game_state == UNKNOWN)
    {
        state->get_legal_actions();
    }
    if (state->game_state == WIN)
    {
        return P_MAX - ply;
    }
    if (state->game_state == DRAW)
    {
        return 0;
    }

    int rep_score;
    if (state->check_repetition(history, rep_score))
    {
        return rep_score;
    }

    int stand_pat = policy_evaluate(state, p, &history);
    if (stand_pat >= beta)
    {
        return stand_pat;
    }
    if (stand_pat > alpha)
    {
        alpha = stand_pat;
    }
    if (ply >= p.max_quiescence_ply)
    {
        return stand_pat;
    }

    history.push(state->hash());
    int best_score = stand_pat;
    auto actions = filter_suicidal_actions(state, ordered_actions(state, true));

    for (auto &action : actions)
    {
        State *next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int raw = same
            ? quiescence(next, alpha, beta, history, ply + 1, ctx, p)
            : quiescence(next, -beta, -alpha, history, ply + 1, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if (score > best_score)
        {
            best_score = score;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        if (alpha >= beta || ctx.stop)
        {
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with alpha-beta pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory &history,
    int ply,
    SearchContext &ctx,
    const MMParams &p)
{
    ctx.nodes++;
    if (ply > ctx.seldepth)
    {
        ctx.seldepth = ply;
    }
    if (ctx.stop)
    {
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if (state->legal_actions.empty() && state->game_state == UNKNOWN)
    {
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    if (state->game_state == WIN)
    {
        return P_MAX - ply;
    }

    if (state->game_state == DRAW)
    {
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if (state->check_repetition(history, rep_score))
    {
        return rep_score;
    }
    history.push(state->hash());

    if (depth <= 0)
    {
        int score = p.use_quiescence
            ? quiescence(state, alpha, beta, history, ply, ctx, p)
            : policy_evaluate(state, p, &history);
        history.pop(state->hash());
        return score;
    }

    /* === Alpha-beta negamax loop === */
    int best_score = M_MAX;
    bool first_child = true;

    auto actions = filter_suicidal_actions(state, ordered_actions(state, false));

    for (auto &action : actions)
    {
        // [Hackathon TODO 3-2 ]
        // create the child state after applying action
        State *next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // [Hackathon TODO 3-3]
        // search the child one level deeper
        int raw;
        if (first_child || !p.use_pvs)
        {
            raw = same
                ? eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p)
                : eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
        }
        else
        {
            raw = same
                ? eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p)
                : eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p);
            int scout_score = same ? raw : -raw;
            if (scout_score > alpha && scout_score < beta)
            {
                raw = same
                    ? eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p)
                    : eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
            }
        }

        // [Hackathon TODO 3-4]
        // convert raw to the current player's perspective.
        int score = same ? raw : -raw;
        first_child = false;

        delete next;

        // [ Hackathon TODO 3-5 ]
        // update best_score if this child is better.
        if (score > best_score)
        {
            best_score = score;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        if (alpha >= beta || ctx.stop)
        {
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}

/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory &history,
    SearchContext &ctx)
{
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if (!state->legal_actions.size())
    {
        state->get_legal_actions();
    }

    if (try_opening_book(state, result))
    {
        return result;
    }

    if (state->game_state == WIN)
    {
        for (auto &action : state->legal_actions)
        {
            if (captures_king(state, action))
            {
                result.best_move = action;
                result.score = P_MAX;
                result.seldepth = ctx.seldepth;
                result.nodes = ctx.nodes;
                result.pv = {result.best_move};
                return result;
            }
        }
    }

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    auto actions = filter_suicidal_actions(state, ordered_actions(state, false));

    for (auto &action : actions)
    {
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */
        State *next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int raw = same
            ? eval_ctx(next, depth - 1, best_score, P_MAX, history, 1, ctx, p)
            : eval_ctx(next, depth - 1, M_MAX, -best_score, history, 1, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if (score > best_score)
        {
            // [ Hackathon TODO 4-2 ]
            // keep this move if it is the best so far
            result.best_move = action;
            best_score = score;

            if (p.report_partial && ctx.on_root_update)
            {
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
    result.score = best_score;
    result.seldepth = ctx.seldepth;
    result.nodes = ctx.nodes;
    result.pv = {result.best_move};

    return result;
}

/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params()
{
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"UsePVS", "true"},
        {"UseQuiescence", "true"},
        {"UsePolicyEndgame", "true"},
        {"MaxQuiescencePly", "4"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs()
{
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
        {"UsePolicyEndgame", ParamDef::CHECK, "true"},
        {"MaxQuiescencePly", ParamDef::SPIN, "4", 0, 30},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
