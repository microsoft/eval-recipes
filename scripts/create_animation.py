# Copyright (c) Microsoft. All rights reserved.

"""
Manim animation script for eval-recipes.

To generate with specific quality:
    uv run manim scripts/create_animation.py EvalRecipesAnimation -qm  # medium quality (720p)
    uv run manim scripts/create_animation.py EvalRecipesAnimation -qh  # high quality (1080p)
    uv run manim scripts/create_animation.py EvalRecipesAnimation -qk  # 4K quality

To generate as GIF:
    uv run manim scripts/create_animation.py EvalRecipesAnimation -qh --format=gif

To save to a specific location:
    uv run manim scripts/create_animation.py EvalRecipesAnimation -o /path/to/output.mp4
"""

from manim import (
    BLACK,
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Code,
    Create,
    CurvedArrow,
    FadeIn,
    Rectangle,
    Scene,
    Table,
    Text,
    VGroup,
    config,
)

config.frame_width = 16
config.frame_height = 9
config.pixel_width = 1920
config.pixel_height = 1080


class EvalRecipesAnimation(Scene):
    def construct(self) -> None:
        title = self.create_title()
        self.show_agent_loop(title)
        self.wait(0.5)

    def create_title(self) -> Text:
        title = Text("eval-recipes", font_size=46, color=BLUE)
        title.to_edge(UP)
        self.play(Create(title), run_time=0.5)
        self.wait(0.2)
        return title

    def show_agent_loop(self, title: Text) -> None:
        """Show the agent loop diagram."""
        # Create rectangles for Messages + Tools and LLM Response
        messages_rect = Rectangle(width=4, height=1.5, color=GREEN, fill_color=GREEN, fill_opacity=0.5)
        messages_text = Text("Messages + Tools", font_size=24, color=BLACK)
        messages_group = VGroup(messages_rect, messages_text)
        messages_text.move_to(messages_rect.get_center())
        messages_group.shift(UP * 0.5)

        llm_rect = Rectangle(width=4, height=1.5, color=YELLOW, fill_color=YELLOW, fill_opacity=0.5)
        llm_text = Text("LLM Response", font_size=24, color=BLACK)
        llm_group = VGroup(llm_rect, llm_text)
        llm_text.move_to(llm_rect.get_center())
        llm_group.shift(DOWN * 1.5)

        # Add "Your agentic loop" text above the rectangles
        step1_text = Text("Your agentic loop", font_size=28, color=WHITE)
        step1_text.next_to(messages_rect, UP, buff=0.5)
        self.play(FadeIn(step1_text), run_time=0.4)

        # Show both rectangles together
        self.play(Create(messages_rect), FadeIn(messages_text), Create(llm_rect), FadeIn(llm_text), run_time=0.5)

        # Create arrows for the loop - connecting closer to corners
        arrow1 = CurvedArrow(
            messages_rect.get_right() + UP * 0.05, llm_rect.get_right() + DOWN * 0.05, angle=-1.2, color=WHITE
        )
        arrow2 = CurvedArrow(
            llm_rect.get_left() + DOWN * 0.05, messages_rect.get_left() + UP * 0.05, angle=-1.2, color=WHITE
        )

        # Animate arrows sequentially - right arrow faster, then left
        self.play(Create(arrow1), run_time=0.4)
        self.play(Create(arrow2), run_time=0.5)
        self.wait(0.3)

        # Slide and shrink everything to the left (including the text)
        agent_loop_group = VGroup(step1_text, messages_rect, messages_text, llm_rect, llm_text, arrow1, arrow2)
        self.play(agent_loop_group.animate.scale(0.7).shift(LEFT * 5.85), run_time=0.75)

        self.show_evaluation_step(messages_rect, llm_rect)
        self.show_scorecard()

    def show_evaluation_step(self, messages_rect: Rectangle, llm_rect: Rectangle) -> None:
        """Show arrows pointing to evaluation rectangle."""
        # Create code block with the Python function signature
        code_text = """async def evaluate(
    messages: ResponseInputParam,
    tools: list[ChatCompletionToolParam],
    evaluations: list[str|type[EvaluatorProtocol]],
) -> list[EvaluationOutput]:"""

        eval_code = Code(
            code_string=code_text,
            language="python",
            background="rectangle",
            tab_width=2,
            add_line_numbers=False,
            paragraph_config={"font_size": 14},
        )
        # Center the evaluation code vertically between the two agent loop rectangles
        eval_code.move_to((-0.25, -0.35, 0))

        # Add descriptive text below the code block
        description_text = Text("Get evaluation results at each step,\nwithout any labels!", font_size=18, color=WHITE)
        description_text.next_to(eval_code, DOWN, buff=0.3)

        # Create arrows from both rectangles to evaluation
        # Arrow from Messages + Tools to evaluate - matching the agent loop arrow style with smaller tip
        tip_length = 0.25
        arrow_to_eval1 = CurvedArrow(
            messages_rect.get_right() + UP * 0.05,
            eval_code.get_left() + UP * 0.05,
            angle=-1.2,
            color=WHITE,
            tip_length=tip_length,
        )
        # Arrow from LLM Response to evaluate - matching the agent loop arrow style with smaller tip
        arrow_to_eval2 = CurvedArrow(
            llm_rect.get_right() + DOWN * 0.05,
            eval_code.get_left() + DOWN * 0.05,
            angle=1.2,
            color=WHITE,
            tip_length=tip_length,
        )

        # Animate everything appearing
        self.play(FadeIn(eval_code), run_time=0.5)
        self.play(FadeIn(description_text), run_time=0.4)
        self.play(Create(arrow_to_eval1), Create(arrow_to_eval2), run_time=0.6)

    def show_scorecard(self) -> None:
        """Show evaluation results scorecard."""
        table_data = [
            ["Evaluation", "Applicable", "Score"],
            ["User Prefs", "Yes", "55%"],
            ["Claim Verification", "Yes", "89%"],
            ["Tool Usage", "Yes", "100%"],
            ["Guidance", "No", "N/A"],
            ["Average", "-", "81%"],
        ]
        scorecard = Table(
            table_data,
            include_outer_lines=False,
            line_config={"stroke_width": 1},
            element_to_mobject_config={"font_size": 20},
            v_buff=0.4,
            h_buff=0.6,
        )

        scorecard.scale(0.6)
        scorecard.shift(RIGHT * 5.7 + DOWN * 0.35)

        title = Text("Turn Scorecard", font_size=24, color=WHITE)
        title.next_to(scorecard, UP, buff=0.25)

        eval_code_right_edge = 2.6  # Move arrow start point left
        arrow_to_results = CurvedArrow(
            (eval_code_right_edge, -0.35, 0),
            scorecard.get_left() + LEFT * 0.15,  # Move arrow end point left
            angle=0,
            color=WHITE,
            tip_length=0.2,
        )

        self.play(FadeIn(title), run_time=0.4)
        self.play(Create(arrow_to_results), run_time=0.5)
        self.play(FadeIn(scorecard), run_time=0.6)
        self.wait(1)


if __name__ == "__main__":
    from manim import tempconfig

    with tempconfig({"quality": "high_quality", "preview": False}):
        scene = EvalRecipesAnimation()
        scene.render()
