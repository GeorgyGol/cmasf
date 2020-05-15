"""Detachmentt of the seasonal components of the time series

    The script contains the function of dividing the time series into a seasonal wave and a trend.
    The dividing occurs according to the algorithm of V. Gubanov (http://www2.forecast.ru/Metodology/Gubanov/VGubanov_05.pdf)
    The function removes additive or multiplacial seasonality, the wave can be dynamic or static

    This file can also be imported as a module and contains the following functions:

    * seasonal_decompose - returns the trend, wave, error and row, cleared from outliers
    * test - retrun pandas DataFrame with seasonal_decompose of testing row
    * neighbours - return cnt_of_neiboors from left-right for cur_index in array

"""

__author__ = "G. Golyshev, V.Gubanov, V. Salnikov"
__copyright__ = "CMASF 2020"
__version__ = "0.0.1"
__maintainer__ = "G. Golyshev"
__email__ = "g.golyshev@forecast.ru"
__status__ = "Production"

import numpy as np

import math
from scipy import stats as st
import cmasf.serv as srv

class DecomposeResult():
    _row=[]
    _wave=[]
    _trend=[]
    _alfa=0
    _std=0
    _steps=0

    def __init__(self, row=[], wave=[], trend=[], alfa=0, std=0, steps=0):
        self._row=row.copy()
        self._wave=wave.copy()
        self._trend=trend.copy()
        self._alfa=alfa
        self._std=std
        self._steps=steps

    @property
    def nobs(self):
        return len(self.row)

    @property
    def observed(self):
        return self._row

    @property
    def trend(self):
        return self._trend

    @property
    def seasonal(self):
        return self._wave

    @property
    def weights(self):
        return self._alfa

    @property
    def std(self):
        return self._std

    @property
    def steps(self):
        return self._steps


class __SeasonWave():
    _row=None
    _period=12
    _static=1
    _model='add'
    _seas_matrix=None

    def __init__(self, source_arr, period=12, model='add', static=1):
        self._model = model

        if model[:3].lower()=='add':
            self._row=source_arr
        elif model[:4].lower()=='mult':
            self._row = np.log(source_arr)
        else:
            raise TypeError('sesonal decompose model undefined')
        self._period=period

        self._static=static
        self._seas_matrix=self._calc_season_matrix()

    def std_trend(self, row):
        err=np.nanstd(row)
        return err

    def std_wave(self, row):
        mtr=srv.as_matrix(row, self._period)
        err=np.nanmean(np.nanstd(mtr, axis=1, ddof=1))#/mtr.shape[0]
        return err

    def _input_model(self, row):
        if self._model[:3].lower()=='add':
            self._row=row
        elif self._model[:4].lower()=='mult':
            self._row = np.log(row)
        else:
            raise TypeError('sesonal decompose model undefined')

    def _output_model(self, wave):
        if self._model[:3] == 'add':
            return self._row - wave, wave, self._row

        elif self._model[:4] == 'mult':
            outr=np.exp(self._row - wave)
            return outr, np.exp(self._row)-outr, np.exp(self._row)
        else:
            raise TypeError('sesonal decompose model undefined')

    def _calc_season_matrix(self):
        """расчет матрицы сезонноси"""
        npbase = np.zeros(self._period)
        npbase[0] = -2
        npbase[-1] = 1
        npbase[1] = 1
        res = [np.roll(npbase, i) for i in range(self._period)]
        res[-1] = np.ones(self._period)
        return np.linalg.inv(np.array(res))

    def _calc_sec_diff(self, row):
        plp = row[-1] * (row[-self._period] / row[-self._period - 1])
        lp = plp * (row[-self._period + 1] / row[-self._period])

        res=srv.as_matrix(np.diff(row, n=2, append=[plp, lp]), period=self._period)
        res[:, -1] = 0
        return res

    def _norm_vect(self, cnt_periods, gamma):
        return np.array([(1 - gamma) / (1 + gamma - gamma ** k - gamma ** (cnt_periods - k + 1)) for k in
                         range(1, cnt_periods + 1)])

    def _weight_matrix(self, row, gamma):
        """правая часть уравнения по Губанову"""
        m_dif2=self._calc_sec_diff(row)
        d_ = np.zeros((m_dif2.shape[1], m_dif2.shape[0]))

        v_weight = self._norm_vect(m_dif2.shape[0], gamma)

        for k in range(m_dif2.shape[0]):
            for j in range(m_dif2.shape[1]):
                # ds_ = np.sum([gamma ** (abs(k - L)) * m_dif2[L, j] for L in range(m_dif2.shape[0])])
                ds_ = np.sum([(gamma ** (abs(k - L))) * m_dif2[L, j] for L in range(m_dif2.shape[0])])
                d_[j, k] = ds_ * v_weight[k]
        return d_

    def _get_wave(self, row, gamma):
        wave = np.ravel(np.matmul(self._seas_matrix, self._weight_matrix(row, gamma)), order='F')
        return np.insert(wave, 0, 0)[:len(row)]

    @property
    def kper(self):
        """количество целых периодов внутри ряда длинной row_lenght"""
        return len(self._row) // self._period

    @property
    def period(self):
        return self._period

    @property
    def last_period_len(self):
        return len(self._row) % self._period

    @property
    def row(self):
        return self._row

    @property
    def season_matrix(self):
        return self._seas_matrix

    @property
    def sec_diff(self):
        return self._calc_sec_diff(self._row)

    def seasX4(self, gamma):
        offs = self.last_period_len
        offs = self._period if offs == 0 else offs
        ins_a = [np.nan] * offs

        w_start = np.append(self._get_wave(self._row[:-offs], gamma), ins_a)
        w_end = np.insert(self._get_wave(self._row[offs:], gamma), 0, ins_a)

        wave = np.nanmean(np.array([w_start, w_end]), axis=0)

        try:
            wave[0] = 2 * w_start[self._period] - w_start[self._period * 2]
        except:
            wave[0] = w_start[self._period]

        return self._output_model(wave)

    def Err_X4(self, trend, wave):
        return self.std_trend(trend)*(1-self._static) + self.std_wave(wave)*self._static

    def Variance(self, gamma):
        trend, wave, _ = self.seasX4(gamma)
        return self.Err_X4(trend, wave)

    def as_matrix(self, fill_val=np.nan):
        # make periods-matrix from row
        return srv.as_matrix(self._row, self._period, fill_val=fill_val)

    def from_matrix(self, src_matrix):
        """very dangerous, must be hiden from outside, using only in seasonal_decompose function"""
        self._row=src_matrix.ravel()[:len(self._row)]
        return self._row

    def outliers_zscore_find(self, src_matrix, pers_minmax_trim=0.2, level_zscore=2.0):
        """find outliers in source time series transformed to period matrix by z-score, return 2D array of indexes"""

        # calc z-score between periods and inside periods

        zscore0 = np.transpose((np.transpose(src_matrix) - st.trim_mean(src_matrix, pers_minmax_trim, axis=1)) / np.mean(
            np.nanstd(src_matrix, axis=1)))
        zscore1 = (src_matrix - st.trim_mean(src_matrix, pers_minmax_trim, axis=0)) / np.mean(np.nanstd(src_matrix, axis=0))

        np.warnings.filterwarnings('ignore')
        x0 = np.abs(zscore0) > level_zscore
        x1 = np.abs(zscore1) > level_zscore
        np.warnings.filterwarnings('default')

        result = np.where(x0 & x1)
        return list(zip(result[0], result[1]))

    def correction_outliers(self, src_matrix, outliers_indexes, corr_by_neibs=3, axis=1):
        """коррекция по соседним точкам периода или через период"""
        def linear_correction(row, outliers_indexes, corr_by_neibs=3):
            """correct outliers by row neiboors"""
            lcor = np.asarray([srv.neighbours(row, i, cnt_of_neiboors=corr_by_neibs,
                                          exclude_from_neibors_index=outliers_indexes) for i in outliers_indexes])
            for i in range(len(outliers_indexes)):
                row[outliers_indexes[i]] = np.nanmean(lcor[i])
            return row

        if axis==1:
            lst_res = []
            for i in range(src_matrix.shape[axis]):
                ind=[k[0] for k in outliers_indexes if k[1]==i]
                lst_res.append( linear_correction(src_matrix[:, i], ind, corr_by_neibs=corr_by_neibs))
            return np.transpose(np.asarray(lst_res))
        elif axis==0:
            lstOut=[y[0] * self.period + y[1] for y in outliers_indexes]
            return linear_correction(self._row, lstOut, corr_by_neibs=corr_by_neibs)
        else:
            # by row neiboors, but with period conuting
            # for i in range(src_matrix.shape[axis]):
            #     ind=[k[1] for k in outliers_indexes if k[0]==i]
            #     lst_res.append( linear_correction(src_matrix[i, :], ind, corr_by_neibs=corr_by_neibs))
            # return np.asarray(lst_res)
            raise(NameError('Correction type not defined'))

def seasonal_decompose(row, period=12, gamma=2, static=0.5, model='additive', precision=0.001,
                       row_correction=False, correction_axis=0, correction_zlevel=2, correction_trimm=0.2,
                       correction_fill_val=np.nan, correction_neiboors=2):

    """Detachmentt of the seasonal components of the time series
        params: row - source row - time series, 1D numpy.array
                periods - points in one period
                gamma - dynamic param for wave varians, if > 1 - function find optimal gamma itself
                static - if =1 the wave will be static, if = 0 - wave will be dynamic, between 0 and 1 - partial static
                model - 'additive' aor 'multiplicative', define wave model
                precision -  precision for gamma calculation if gamma calculating itself
                row_correction - if True make outliers row correction
                correction_axis - find outliers and correct: 0 - by flat row, 1 - by inter-period
                correction_zlevel - z-score level for outlier point
                correction_trimm - exclude min-max pointer from find correction alg., in percent
                correction_fill_val - fill row with this value for make matrix
                correction_neiboors - correct ouliers by neiboors's mean, this param - count for used neiboors
                pow_weight - power wight, more 1 - smoother trend

        return: DecomposeResult

        example: res = seasonal_decompose(row, period=12, gamma=2, static=0.1, model='add')
    """
    x=__SeasonWave(row.copy(), period=period, static=static, model=model)
    if row_correction:
        xmtr = x.as_matrix(fill_val=correction_fill_val)
        out_indexes=x.outliers_zscore_find(xmtr, pers_minmax_trim=correction_trimm, level_zscore=correction_zlevel)
        x.from_matrix(x.correction_outliers(xmtr,  out_indexes,  corr_by_neibs=correction_neiboors, axis=correction_axis))

    if 0 < gamma <= 1:
        trend, wave, out_row = x.seasX4(gamma)
        err = x.Err_X4(trend, wave)
        ret = DecomposeResult(row=out_row, trend=trend, wave=wave, std=err, alfa=gamma, steps=1)
        return ret

    r = (3 - math.sqrt(5)) / 2
    A = 0
    c = 1
    k = 0

    while k < 30:
        B = A + r * (c - A)
        d = c + r * (A - c)

        err1 = x.Variance(d)
        err2 = x.Variance(B)

        if abs(1 - err1 / err2) < precision: break

        if err1 < err2:
            A = B
        else:
            c = A
            A = d
        k += 1

    alfa = d if err1 < err2 else B

    trend, wave, out_row = x.seasX4(alfa)
    err = x.Err_X4(trend, wave)
    ret=DecomposeResult(row=out_row, trend=trend, wave=wave, std=err, alfa=alfa, steps=k)
    return ret

# def test():
#     # inp= np.array( (305.6, 296.6, 286.1, 303.4, 287.5, 293.4, 286.2, 292, 290, 285.8, 307.8, 302.1,
#     #                 311, 301, 280.7, 311, 299.1, 307.7, 297.3, 299.4, 302.5, 299.9, 326.9, 313.7, 311.3, 311.4, 297.9,
#     #                 328, 315.6, 323.5, 312, 321.3, 321.7, 322, 341.7, 331.2, 341.7, 340.2, 315.7, 353.1, 336.5, 347.7,
#     #                 339.8, 346.5, 350.2, 342.5, 367.4, 357.5, 373.9, 362.3, 345.4, 376.6, 367.1, 371.6, 357.8, 366.1,
#     #                 373.1, 366.1, 380.2, 375.7325, 386.6126) ) #np.around(np.random.random(24), 2)
#
#     inp = np.array((305.6, 396.6, 386.1, 303.4, 487.5, 293.4, 286.2, 292, 290, 285.8, 307.8, 302.1,
#                      211, 301, 280.7, 311, 299.1, 307.7, 297.3, 299.4, 302.5, 299.9, 326.9, 313.7, 311.3, 311.4, 297.9,
#                      328, 315.6, 323.5, 212, 321.3, 221.7, 322, 341.7, 331.2, 341.7, 340.2, 315.7, 353.1, 336.5, 347.7,
#                      339.8, 346.5, 550.2, 342.5, 367.4, 357.5, 373.9, 362.3, 345.4, 376.6, 367.1, 371.6, 357.8, 366.1,
#                      373.1, 366.1, 380.2, 375.7325, 386.6126) )  # np.around(np.random.random(24), 2)
#
#     np.set_printoptions(precision=3, suppress=True, linewidth =110)
#     row = inp[:-2]
#     trend, wave, err, out_row = seasonal_decompose(row.copy(), period=12, gamma=2, static=0, model='add',
#                                                    correction_zlevel=1.1,
#                                                    row_correction=True, correction_axis=1, correction_neiboors=3)
#     pdf = pd.DataFrame({'row': row, 'wave': wave, 'trend': trend, 'corr_row':out_row})
#
#     return pdf


if __name__ == "__main__":
    # inp = np.array((305.6, 296.6, 286.1, 303.4, 287.5, 293.4, 286.2, 292, 290, 285.8, 307.8, 302.1,
    #                 311, 301, 380.7, 311, 199.1, 307.7, 297.3, 299.4, 302.5, 299.9, 326.9, 113.7, 311.3, 311.4, 97.9,
    #                 328, 315.6, 323.5, 312, 321.3, 321.7, 322, 341.7, 331.2, 341.7, 340.2, 315.7, 353.1, 336.5, 347.7,
    #                 139.8, 346.5, 350.2, 142.5, 0, 357.5, 373.9, 362.3, 345.4, 376.6, 367.1, 371.6, 357.8, 366.1,
    #                 373.1, 366.1, 380.2, 375.7325, 386.6126))  # np.around(np.random.random(24), 2)
    #
    # def calc(i):
    #     x=np.roll(inp, shift=-i)
    #     return x[0]-2*x[1]+x[2]
    #
    # np.set_printoptions(precision=3, suppress=True, linewidth =110)
    #
    #
    # # x = __SeasonWave(inp, period=12, static=0.1, model='add')
    # #
    # # plp = inp[-1] * (inp[-x.period] / inp[-x.period - 1])
    # # lp = plp * (inp[-x.period + 1] / inp[-x.period])
    # #
    # # inp=np.append(inp, [plp, lp])
    # #
    # # print(inp)
    # #
    # # mtr=np.asarray([calc(i) for i in range(len(inp))][:60])
    # # mtr=mtr.reshape( (x.kper, x.period) )
    # # mtr[:, -1]=0
    # # print(mtr)
    # # xm=x.calc_sec_diff(inp)
    # # print(xm)
    # #
    # # x = _SeasonWave(inp, period=12, static=0.1, model='add')
    # # print(x.row)
    # # xmtr=x.as_matrix()
    # # print(xmtr)
    #
    # p=test()
    #
    # p.plot.line()
    # plt.show()

    print('Hello from CMASF seasonal_decompose')
