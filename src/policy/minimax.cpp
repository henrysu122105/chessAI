#include <algorithm>
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
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(state->hash());
        return score;
    }

    /* === Alpha-beta negamax loop === */
    int best_score = M_MAX;

    auto actions = state->legal_actions;
    std::sort(actions.begin(), actions.end(), [state](const Move &a, const Move &b) {
        return move_order_score(state, a) > move_order_score(state, b);
    });

    for (auto &action : actions)
    {
        // [Hackathon TODO 3-2 ]
        // create the child state after applying action
        State *next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // [Hackathon TODO 3-3]
        // search the child one level deeper
        int raw = same
            ? eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p)
            : eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);

        // [Hackathon TODO 3-4]
        // convert raw to the current player's perspective.
        int score = same ? raw : -raw;

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

    auto actions = state->legal_actions;
    std::sort(actions.begin(), actions.end(), [state](const Move &a, const Move &b) {
        return move_order_score(state, a) > move_order_score(state, b);
    });

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
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs()
{
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
