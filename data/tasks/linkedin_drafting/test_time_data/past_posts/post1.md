# What Cicadas Taught Me About Caching

Every 17 years, billions of cicadas emerge from underground in perfect synchronization. They've been down there the entire time, following their own ancient clock. Then one spring, they all surface at once.

This is exactly how I think about cache invalidation.

Most engineers set cache TTLs the same way for everything—60 seconds, 5 minutes, whatever feels right. But different data has fundamentally different rhythms.

**Some data is like mayflies.** User session state, real-time prices, "who's online now"—this needs to refresh constantly. TTL measured in seconds. Cache it, but barely.

**Some data is like annual crops.** Product catalogs, team rosters, configuration values—this changes on human schedules. Daily or weekly updates are fine. Cache aggressively, invalidate deliberately.

**Some data is like cicadas.** Database schemas, API contracts, core business rules—this changes on geological timescales. When it does change, it's a major event. Cache almost forever, but have a plan for the rare emergence.

The problem isn't caching—it's that we treat everything like it updates at the same frequency. We're afraid to cache too long, so we cache too little.

I once worked with a system that refetched our pricing rules every 30 seconds. These rules changed maybe twice a year, with weeks of planning each time. We were wasting thousands of database queries per minute to "stay fresh" on data that had a 6-month update cycle.

**The art isn't picking a TTL. It's understanding the natural rhythm of your data.** What actually changes hourly vs. daily vs. quarterly? Match your caching strategy to reality, not to fear.

Next time you write `cache.set(key, value, 300)`, ask yourself: Is this data a mayfly, a crop, or a cicada?

What's the longest cache TTL you've ever used in production?
