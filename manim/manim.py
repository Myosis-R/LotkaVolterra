from manim import *
class Flow(MovingCameraScene):
    def construct(self):
        ax = Axes(x_range=[-3,3],y_range=[-2,2],x_length=6,y_length=4).add_coordinates()
        self.add(ax)

        func = lambda pos: pos[0]*(-1+pos[0])*RIGHT+pos[1]*(-2*pos[0]+1+pos[1])*UP
        funcColor = lambda pos: np.log(np.sum(func(pos)**2))
        
        self.camera.frame.set(height=5).shift(RIGHT)
        
        stream_lines = StreamLines(
                func,
                color_scheme=funcColor,
                min_color_scheme_value=0,
                max_color_scheme_value=10,
                x_range=[-3,3,0.1],
                y_range=[-2,2,0.1],
                stroke_width=1,
                max_anchors_per_line=10,
                virtual_time=3,
                padding=1,
                )
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False,flow_speed=1.5)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
