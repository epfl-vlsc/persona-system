# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Student t distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from scipy import stats
import tensorflow as tf


class StudentTTest(tf.test.TestCase):

  def testStudentPDFAndLogPDF(self):
    with tf.Session():
      batch_size = 6
      df = tf.constant([3.0] * batch_size)
      mu = tf.constant([7.0] * batch_size)
      sigma = tf.constant([8.0] * batch_size)
      df_v = 3.0
      mu_v = 7.0
      sigma_v = 8.0
      t = np.array([-2.5, 2.5, 8.0, 0.0, -1.0, 2.0], dtype=np.float32)
      student = tf.contrib.distributions.StudentT(df, mu=mu, sigma=sigma)

      log_pdf = student.log_pdf(t)
      self.assertEquals(log_pdf.get_shape(), (6,))
      log_pdf_values = log_pdf.eval()
      pdf = student.pdf(t)
      self.assertEquals(pdf.get_shape(), (6,))
      pdf_values = pdf.eval()

      expected_log_pdf = stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
      expected_pdf = stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllClose(np.log(expected_pdf), log_pdf_values)
      self.assertAllClose(expected_pdf, pdf_values)
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentLogPDFMultidimensional(self):
    with tf.Session():
      batch_size = 6
      df = tf.constant([[1.5, 7.2]] * batch_size)
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      df_v = np.array([1.5, 7.2])
      mu_v = np.array([3.0, -3.0])
      sigma_v = np.array([np.sqrt(10.0), np.sqrt(15.0)])
      t = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      student = tf.contrib.distributions.StudentT(df, mu=mu, sigma=sigma)
      log_pdf = student.log_pdf(t)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      pdf = student.pdf(t)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      expected_log_pdf = stats.t.logpdf(t, df_v, loc=mu_v, scale=sigma_v)
      expected_pdf = stats.t.pdf(t, df_v, loc=mu_v, scale=sigma_v)
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllClose(np.log(expected_pdf), log_pdf_values)
      self.assertAllClose(expected_pdf, pdf_values)
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testStudentEntropy(self):
    df_v = np.array([[2., 3., 7.]])  # 1x3
    mu_v = np.array([[1., -1, 0]])  # 1x3
    sigma_v = np.array([[1., 2., 3.]]).T  # transposed => 3x1
    with tf.Session():
      student = tf.contrib.distributions.StudentT(df=df_v,
                                                  mu=mu_v,
                                                  sigma=sigma_v)
      ent = student.entropy()
      ent_values = ent.eval()

    # Help scipy broadcast to 3x3
    ones = np.array([[1, 1, 1]])
    sigma_bc = sigma_v * ones
    mu_bc = ones.T * mu_v
    df_bc = ones.T * df_v
    expected_entropy = stats.t.entropy(
        np.reshape(df_bc, [-1]),
        loc=np.reshape(mu_bc, [-1]),
        scale=np.reshape(sigma_bc, [-1]))
    expected_entropy = np.reshape(expected_entropy, df_bc.shape)
    self.assertAllClose(expected_entropy, ent_values)

  def testStudentSample(self):
    with tf.Session():
      df = tf.constant(4.0)
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(10.0))
      df_v = 4.0
      mu_v = 3.0
      sigma_v = np.sqrt(10.0)
      n = tf.constant(100000)
      student = tf.contrib.distributions.StudentT(df=df, mu=mu, sigma=sigma)
      samples = student.sample(n, seed=137)
      sample_values = samples.eval()
      n = 100000
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(sample_values.mean(), mu_v, atol=1e-2)
      self.assertAllClose(sample_values.var(),
                          sigma_v**2 * df_v / (df_v - 2),
                          atol=.25)
      self._checkKLApprox(df_v, mu_v, sigma_v, sample_values)

  def testStudentSampleMultiDimensional(self):
    with tf.Session():
      batch_size = 7
      df = tf.constant([[3.0, 7.0]] * batch_size)
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      df_v = [3.0, 7.0]
      mu_v = [3.0, -3.0]
      sigma_v = [np.sqrt(10.0), np.sqrt(15.0)]
      n = tf.constant(100000)
      student = tf.contrib.distributions.StudentT(df=df, mu=mu, sigma=sigma)
      samples = student.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (100000, batch_size, 2))
      self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=.15)
      self.assertAllClose(sample_values[:, 0, 0].var(),
                          sigma_v[0]**2 * df_v[0] / (df_v[0] - 2),
                          atol=1)
      self._checkKLApprox(df_v[0], mu_v[0], sigma_v[0], sample_values[:, 0, 0])
      self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=.01)
      self.assertAllClose(sample_values[:, 0, 1].var(),
                          sigma_v[1]**2 * df_v[1] / (df_v[1] - 2),
                          atol=.25)
      self._checkKLApprox(df_v[0], mu_v[0], sigma_v[0], sample_values[:, 0, 1])

  def _checkKLApprox(self, df, mu, sigma, samples):
    n = samples.size
    np.random.seed(137)
    sample_scipy = stats.t.rvs(df, loc=mu, scale=sigma, size=n)
    covg = 0.99
    r = stats.t.interval(covg, df, loc=mu, scale=sigma)
    bins = 100
    hist, _ = np.histogram(samples, bins=bins, range=r)
    hist_scipy, _ = np.histogram(sample_scipy, bins=bins, range=r)
    self.assertGreater(hist.sum(), n * (covg - .01))
    self.assertGreater(hist_scipy.sum(), n * (covg - .01))
    hist_min1 = hist + 1.  # put at least one item in each bucket
    hist_norm = hist_min1 / hist_min1.sum()
    hist_scipy_min1 = hist_scipy + 1.  # put at least one item in each bucket
    hist_scipy_norm = hist_scipy_min1 / hist_scipy_min1.sum()
    kl_appx = np.sum(np.log(hist_scipy_norm / hist_norm) * hist_scipy_norm)
    self.assertLess(kl_appx, 1)

  def testBroadcastingParams(self):

    def _check(student):
      self.assertEqual(student.mean().get_shape(), (3,))
      self.assertEqual(student.variance().get_shape(), (3,))
      self.assertEqual(student.entropy().get_shape(), (3,))
      self.assertEqual(student.log_pdf(2.).get_shape(), (3,))
      self.assertEqual(student.pdf(2.).get_shape(), (3,))
      self.assertEqual(student.sample(37).get_shape(), (37, 3,))

    _check(tf.contrib.distributions.StudentT(df=[2., 3., 4.,], mu=2., sigma=1.))
    _check(tf.contrib.distributions.StudentT(df=7., mu=[2., 3., 4.,], sigma=1.))
    _check(tf.contrib.distributions.StudentT(df=7., mu=3., sigma=[2., 3., 4.,]))

  def testBroadcastingPdfArgs(self):

    def _assert_shape(student, arg, shape):
      self.assertEqual(student.log_pdf(arg).get_shape(), shape)
      self.assertEqual(student.pdf(arg).get_shape(), shape)

    def _check(student):
      _assert_shape(student, 2., (3,))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (3,))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check(tf.contrib.distributions.StudentT(df=[2., 3., 4.,], mu=2., sigma=1.))
    _check(tf.contrib.distributions.StudentT(df=7., mu=[2., 3., 4.,], sigma=1.))
    _check(tf.contrib.distributions.StudentT(df=7., mu=3., sigma=[2., 3., 4.,]))

    def _check2d(student):
      _assert_shape(student, 2., (1, 3))
      xs = np.array([2., 3., 4.], dtype=np.float32)
      _assert_shape(student, xs, (1, 3))
      xs = np.array([xs])
      _assert_shape(student, xs, (1, 3))
      xs = xs.T
      _assert_shape(student, xs, (3, 3))

    _check2d(tf.contrib.distributions.StudentT(
        df=[[2., 3., 4.,]], mu=2., sigma=1.))
    _check2d(tf.contrib.distributions.StudentT(
        df=7., mu=[[2., 3., 4.,]], sigma=1.))
    _check2d(tf.contrib.distributions.StudentT(
        df=7., mu=3., sigma=[[2., 3., 4.,]]))

    def _check2d_rows(student):
      _assert_shape(student, 2., (3, 1))
      xs = np.array([2., 3., 4.], dtype=np.float32)  # (3,)
      _assert_shape(student, xs, (3, 3))
      xs = np.array([xs])  # (1,3)
      _assert_shape(student, xs, (3, 3))
      xs = xs.T  # (3,1)
      _assert_shape(student, xs, (3, 1))

    _check2d_rows(tf.contrib.distributions.StudentT(
        df=[[2.], [3.], [4.]], mu=2., sigma=1.))
    _check2d_rows(tf.contrib.distributions.StudentT(
        df=7., mu=[[2.], [3.], [4.]], sigma=1.))
    _check2d_rows(tf.contrib.distributions.StudentT(
        df=7., mu=3., sigma=[[2.], [3.], [4.]]))

  def testMean(self):
    with tf.Session():
      mu = [-2, 0., 1., 3.3, 4.4]
      student = tf.contrib.distributions.StudentT(
          df=[0.5, 1., 3., 5., 7.],
          mu=mu,
          sigma=[5., 4., 3., 2., 1.])
      # Test broadcast of mu across shape of df/sigma
      mean = student.mean().eval()
      self.assertAllClose([np.nan, np.nan, 1., 3.3, 4.4], mean)

  def testVariance(self):
    with tf.Session():
      df = [0.5, 1., 3., 5., 7.]
      mu = [-2, 0., 1., 3.3, 4.4]
      sigma = [5., 4., 3., 2., 1.]
      student = tf.contrib.distributions.StudentT(df=df, mu=mu, sigma=sigma)
      # Test broadcast of mu across shape of df/sigma
      var = student.variance().eval()
      # scipy uses inf rather than nan here.  Assert we use NaN, then replace
      # with infinity to compare to scipy.
      self.assertFalse(np.isinf(var).any())
      var[np.isnan(var)] = np.inf

      expected_var = [
          stats.t.var(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)]
      self.assertAllClose(expected_var, var)

  def testStd(self):
    with tf.Session():
      df = [0.5, 1., 3., 5., 7.]
      mu = [-2, 0., 1., 3.3, 4.4]
      sigma = [5., 4., 3., 2., 1.]
      student = tf.contrib.distributions.StudentT(df=df, mu=mu, sigma=sigma)
      # Test broadcast of mu across shape of df/sigma
      std = student.std().eval()
      # scipy uses inf rather than nan here.  Assert we use NaN, then replace
      # with infinity to compare to scipy.
      self.assertFalse(np.isinf(std).any())
      std[np.isnan(std)] = np.inf

      expected_std = [
          stats.t.std(d, loc=m, scale=s) for (d, m, s) in zip(df, mu, sigma)]
      self.assertAllClose(expected_std, std)

  def testMode(self):
    with tf.Session():
      student = tf.contrib.distributions.StudentT(
          df=[0.5, 1., 3],
          mu=[-1, 0., 1],
          sigma=[5., 4., 3.])
      # Test broadcast of mu across shape of df/sigma
      mode = student.mode().eval()
      self.assertAllClose([-1., 0, 1], mode)

  def testPdfOfSample(self):
    with tf.Session() as sess:
      student = tf.contrib.distributions.StudentT(df=3., mu=np.pi, sigma=1.)
      num = 20000
      samples = student.sample(num, seed=137)
      pdfs = student.pdf(samples)
      mean = student.mean()
      mean_pdf = student.pdf(student.mean())
      sample_vals, pdf_vals, mean_val, mean_pdf_val = sess.run(
          [samples, pdfs, student.mean(), mean_pdf])
      self.assertEqual(samples.get_shape(), (num,))
      self.assertEqual(pdfs.get_shape(), (num,))
      self.assertEqual(mean.get_shape(), ())
      self.assertNear(np.pi, np.mean(sample_vals), err=0.02)
      self.assertNear(np.pi, mean_val, err=1e-6)
      self.assertNear(stats.t.pdf(np.pi, 3., loc=np.pi), mean_pdf_val, err=1e-6)
      # Verify integral over sample*pdf ~= 1.
      self._assertIntegral(sample_vals, pdf_vals)

  def testPdfOfSampleMultiDims(self):
    with tf.Session() as sess:
      student = tf.contrib.distributions.StudentT(df=[7., 11.],
                                                  mu=[[5.], [6.]],
                                                  sigma=3.)
      num = 50000
      samples = student.sample(num, seed=137)
      pdfs = student.pdf(samples)
      sample_vals, pdf_vals = sess.run([samples, pdfs])
      self.assertEqual(samples.get_shape(), (num, 2, 2))
      self.assertEqual(pdfs.get_shape(), (num, 2, 2))
      self.assertNear(5., np.mean(sample_vals[:, 0, :]), err=.03)
      self.assertNear(6., np.mean(sample_vals[:, 1, :]), err=.03)
      self.assertNear(stats.t.var(7., loc=0., scale=3.),  # loc d.n. effect var
                      np.var(sample_vals[:, :, 0]),
                      err=.25)
      self.assertNear(stats.t.var(11., loc=0., scale=3.),  # loc d.n. effect var
                      np.var(sample_vals[:, :, 1]),
                      err=.25)
      self._assertIntegral(sample_vals[:, 0, 0], pdf_vals[:, 0, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 0, 1], pdf_vals[:, 0, 1], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 0], pdf_vals[:, 1, 0], err=0.02)
      self._assertIntegral(sample_vals[:, 1, 1], pdf_vals[:, 1, 1], err=0.02)

  def _assertIntegral(self, sample_vals, pdf_vals, err=1e-3):
    s_p = zip(sample_vals, pdf_vals)
    prev = (sample_vals.min() - 1000, 0)
    total = 0
    for k in sorted(s_p, key=lambda x: x[0]):
      pair_pdf = (k[1] + prev[1]) / 2
      total += (k[0] - prev[0]) * pair_pdf
      prev = k
    self.assertNear(1., total, err=err)

  def testNegativeDofFails(self):
    with tf.Session():
      student = tf.contrib.distributions.StudentT(df=[2, -5.],
                                                  mu=0.,
                                                  sigma=1.,
                                                  name='S')
      with self.assertRaisesOpError(r'Condition x > 0 did not hold'):
        student.mean().eval()

  def testNegativeScaleFails(self):
    with tf.Session():
      student = tf.contrib.distributions.StudentT(df=[5.],
                                                  mu=0.,
                                                  sigma=[[3.], [-2.]],
                                                  name='S')
      with self.assertRaisesOpError(r'Condition x > 0 did not hold'):
        student.mean().eval()


if __name__ == '__main__':
  tf.test.main()
