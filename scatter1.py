from visdom import Visdom
import numpy as np
viz = Visdom()


def  plotScatter(X,Y):
        # 3d scatterplot with custom labels and ranges
        viz.scatter(
            X=np.random.rand(100, 3),
            Y=(Y + 1.5).astype(int),
            opts=dict(
                legend=['Men', 'Women'],
                markersize=5,
                xtickmin=0,
                xtickmax=2,
                xlabel='Arbitrary',
                xtickvals=[0, 0.75, 1.6, 2],
                ytickmin=0,
                ytickmax=2,
                ytickstep=0.5,
                ztickmin=0,
                ztickmax=1,
                ztickstep=0.5,
            )
        )



if __name__ == '__main__':

    Y = np.random.rand(100)
    plotScatter(X, Y)
