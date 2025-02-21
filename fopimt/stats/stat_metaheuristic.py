from .stat import Stat
from ..solutions.solution import Solution
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import os


class StatMetaheuristic(Stat):
    """
    Statistical module for metaheuristic algorithms. Compute descriptive statistics and pairwise comparison and
    ranking.
    """
    def _init_params(self):
        super()._init_params()
        self._data = None

    ####################################################################
    #########  Public functions
    ####################################################################
    def evaluate_statistic(self, solutions: list[Solution]):
        df = self._organize_results(solutions)
        self._data = self._get_statistics(df)

    def export(self, path: str):

        results, rank_dims, df_pairwise, desc_stats, df_rank_all = self._data

        # Results on specific function
        for r in results:
            fun = r['func']
            dim = r['dim']
            result = r['result']
            filename = 'S_F_' + fun + '_D_' + str(dim) + '.csv'
            result.to_csv(os.path.join(path, filename), index=True)

        # Ranking over one dimension
        for rd in rank_dims:
            dim = rd['dim']
            df_rank_dim = rd['df_rank_dim']
            filename = 'S_ranking_D_' + str(dim) + '.csv'
            df_rank_dim.to_csv(os.path.join(path, filename), index=True)

        # Pair-wise results output
        filename = 'S_pairwise.csv'
        df_pairwise.to_csv(os.path.join(path, filename))

        # Descriptive statistics on all functions in all dimensions
        filename = 'S_descriptive.csv'
        desc_stats.to_csv(os.path.join(path, filename), index=False)

        # Ranking over all test functions in all dimensions
        filename = 'S_ranking.csv'
        df_rank_all.to_csv(os.path.join(path, filename), index=True)

    @classmethod
    def get_short_name(cls) -> str:
        return "stat.meta"

    @classmethod
    def get_long_name(cls) -> str:
        return "Metaheuristic"

    @classmethod
    def get_description(cls) -> str:
        return "Statistical module for metaheuristic algorithms. Compute descriptive statistics and pairwise " \
                      "comparison and ranking."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': {'metaheuristic'},
            'output': set()
        }

    ####################################################################
    #########  Private functions
    ####################################################################

    def _organize_results(self, solutions: list[Solution]) -> pd.DataFrame:
        """
        Parser function. Transfer results from Solution.metadata to pd.DataFrame suitable for statistical evaluation.
        :param solutions: List of Solutions.
        :return: pd.DataFrame with data from provided Solutions
        """
        columns = ['algorithm', 'function', 'dim', 'ofvs', 'params']
        df = pd.DataFrame(columns=columns)
        solutions_filtered = []
        for solution in solutions:
            if 'results' in solution.get_metadata().keys():
                solutions_filtered.append(solution)

        for solution in solutions_filtered:
            alg = os.path.basename(solution.get_path()).split('.')[0]
            df_metadata: pd.DataFrame = solution.get_metadata()['results']
            for index, row in df_metadata.iterrows():
                fun = row['Function']
                dim = row['Dimension']
                ofvs = row['Result_fitness']
                params = row['Result_params']
                new_row = {'algorithm': alg, 'function': fun, 'dim': dim, 'ofvs': ofvs, 'params': params}
                df.loc[len(df)] = new_row

        return df

    def _get_statistics(self, df: pd.DataFrame, alpha=0.05):
        """
        Compute the statistical data.
        :param df: DataFrame source.
        :param alpha: Statistical significance level. Default value 0.05
        :return: zip[results, rank_dims, df_pairwise, desc_stats, df_rank_all]
        """
        algorithms = df['algorithm'].unique()
        dims = df['dim'].unique()
        functions = df['function'].unique()

        columns = ['algorithm', 'function', 'dim', 'min', 'max', 'median', 'mean', 'std', 'ranking']
        desc_stats = pd.DataFrame(columns=columns)

        res_columns = np.concatenate((algorithms, ['score', 'ranking']))

        df_rank_all = pd.DataFrame(index=algorithms, columns=['score', 'ranking'])
        dict_rank_all = {}
        for a in algorithms:
            dict_rank_all[a] = 0

        # Dataframe for pair-wise results
        df_pairwise = pd.DataFrame(0, index=algorithms, columns=algorithms)

        results = []
        rank_dims = []

        for dim in dims:

            df_rank_dim = pd.DataFrame(index=algorithms, columns=['score', 'ranking'])
            dict_rank_dim = {}
            for a in algorithms:
                dict_rank_dim[a] = 0

            for fun in functions:

                result = pd.DataFrame(index=algorithms, columns=res_columns)

                for row_player in algorithms:

                    print(row_player, fun, dim)
                    rp_result = \
                        list(
                            df[(df['algorithm'] == row_player) & (df['function'] == fun) & (df['dim'] == dim)]['ofvs'])[
                            0]

                    new_row = {'algorithm': row_player, 'function': fun, 'dim': dim, 'min': min(rp_result),
                               'max': max(rp_result), 'median': np.median(rp_result), 'mean': np.mean(rp_result),
                               'std': np.std(rp_result), 'ranking': 0.0}

                    desc_stats.loc[len(desc_stats)] = new_row
                    result.loc[row_player, 'score'] = 0

                    for col_player in algorithms:
                        if row_player == col_player:
                            result.loc[row_player, col_player] = '0'  # No match against themselves
                        else:
                            # Result
                            rp_result = list(
                                df[(df['algorithm'] == row_player) & (df['function'] == fun) & (df['dim'] == dim)][
                                    'ofvs'])[
                                0]
                            cp_result = list(
                                df[(df['algorithm'] == col_player) & (df['function'] == fun) & (df['dim'] == dim)][
                                    'ofvs'])[
                                0]

                            stat, p_value = ranksums(rp_result, cp_result)

                            # Interpret the results
                            if p_value < alpha:
                                if stat < 0:
                                    sign = '+'
                                    result.loc[row_player, 'score'] += 1
                                    dict_rank_all[row_player] += 1
                                    dict_rank_dim[row_player] += 1
                                    df_pairwise.loc[row_player, col_player] += 1
                                else:
                                    sign = '-'
                                    result.loc[row_player, 'score'] -= 1
                                    dict_rank_all[row_player] -= 1
                                    dict_rank_dim[row_player] -= 1
                                    df_pairwise.loc[row_player, col_player] -= 1
                            else:
                                sign = '='

                            result.loc[row_player, col_player] = sign  # Random result for illustration

                result['ranking'] = result['score'].rank(method='average', ascending=False)
                result = result.sort_values(by='ranking')
                # Get algorithm name and its ranking
                for index, _ in result.iterrows():
                    desc_stats.loc[(desc_stats['algorithm'] == index) & (desc_stats['function'] == fun) & (
                            desc_stats['dim'] == dim), 'ranking'] = result.loc[index, 'ranking']

                results.append({
                    'dim': dim,
                    'func': fun,
                    'result': result
                })

            for a in algorithms:
                df_rank_dim.loc[a, 'score'] = dict_rank_dim[a]

            df_rank_dim['ranking'] = df_rank_dim['score'].rank(method='average', ascending=False)
            df_rank_dim = df_rank_dim.sort_values(by='ranking')

            rank_dims.append(
                {
                    'dim': dim,
                    'df_rank_dim': df_rank_dim
                }
            )

        for a in algorithms:
            df_rank_all.loc[a, 'score'] = dict_rank_all[a]

        df_rank_all['ranking'] = df_rank_all['score'].rank(method='average', ascending=False)
        df_rank_all = df_rank_all.sort_values(by='ranking')

        return results, rank_dims, df_pairwise, desc_stats, df_rank_all
