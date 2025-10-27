# Your Microservices Are a Medieval City

I visited Carcassonne last summer. It's a perfectly preserved medieval walled city in France, and walking through it, I couldn't stop thinking about microservices.

Medieval cities had walls for defense. But they couldn't seal themselves off completely—they needed trade. So they built gates. Controlled entry points. Guards who checked what came in and out.

Inside the walls, different districts had autonomy. The blacksmiths' quarter, the merchants' district, the religious center. Each area had its own rules, its own governance. But they all shared the city's infrastructure—the wells, the walls, the gates.

**This is exactly how microservices should work.**

**Your service boundaries are walls.** They protect your internal logic from external chaos. When a dependency goes down, your walls keep the damage contained. But like medieval walls, they only work if you actually enforce them.

**Your APIs are gates.** Not every door is a gate. Medieval cities had one or two main gates, not a hundred random holes in the wall. Your service should expose intentional APIs, not accidental coupling through shared databases or message queues that become de facto APIs.

**Your teams are districts.** The blacksmiths didn't ask the merchants for permission to reorganize their shops. Districts had autonomy within their walls. That's what "independently deployable" actually means—not just technically possible, but organizationally expected.

The problem I see constantly: teams build the walls but forget the gates. Or they build so many gates the walls become useless. Or they build walls but then tunnel secret passages underneath them (looking at you, shared database).

**Medieval cities worked because they balanced isolation and integration.** Too isolated and the city starved. Too integrated and defense became impossible. Same with your services.

When you draw that next service boundary, ask: Am I building a wall, a gate, or a secret tunnel?

What's the worst "secret tunnel" you've seen between services?
