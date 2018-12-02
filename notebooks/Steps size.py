# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

# +
import sys

sys.path = sys.path + ['../']
# -


from handcut_mcmc import mcmc

data = np.random.randn(20)


posterior = mcmc.sampler(data, samples=1000)

plt.plot(posterior)

import seaborn as sns

sns.distplot(posterior[100:])


