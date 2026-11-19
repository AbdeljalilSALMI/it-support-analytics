-- 1. Ticket demand by category.
SELECT category, COUNT(*) AS total_tickets
FROM tickets
GROUP BY category
ORDER BY total_tickets DESC, category;

-- 2. Typical and tail delays by priority, with outcome coverage.
SELECT priority, COUNT(*) AS total_tickets,
       COUNT(resolution_wallclock_hours) AS tickets_with_duration,
       ROUND(AVG(resolution_wallclock_hours)::numeric, 2) AS mean_hours,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_wallclock_hours))::numeric, 2) AS median_hours,
       ROUND((PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_wallclock_hours))::numeric, 2) AS p90_hours
FROM tickets
GROUP BY priority
ORDER BY median_hours DESC NULLS LAST, priority;

-- 3. Workload and observed outcomes by assigned support team.
SELECT assigned_team, COUNT(*) AS total_tickets,
       COUNT(resolution_wallclock_hours) AS tickets_with_duration
FROM tickets
GROUP BY assigned_team
ORDER BY total_tickets DESC, assigned_team;

-- 4. Monthly intake and recorded-duration coverage by creation cohort.
SELECT DATE_TRUNC('month', created_at)::date AS creation_month,
       COUNT(*) AS total_tickets,
       ROUND(100.0 * COUNT(resolution_wallclock_hours) / COUNT(*), 2) AS recorded_duration_pct,
       ROUND(AVG(resolution_wallclock_hours)::numeric, 2) AS mean_hours
FROM tickets
GROUP BY 1
ORDER BY 1;

-- 5. Slow outcome share among labeled tickets by category.
SELECT category, COUNT(is_fast_resolution) AS labeled_tickets,
       COUNT(*) FILTER (WHERE is_fast_resolution IS FALSE) AS slow_tickets,
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_fast_resolution IS FALSE)
             / NULLIF(COUNT(is_fast_resolution), 0), 2) AS slow_pct
FROM tickets
GROUP BY category
ORDER BY slow_pct DESC NULLS LAST, category;

-- 6. Recurring issue types with the longest median recorded duration.
SELECT issue_type_id, COUNT(resolution_wallclock_hours) AS tickets_with_duration,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_wallclock_hours))::numeric, 2) AS median_hours
FROM tickets
GROUP BY issue_type_id
HAVING COUNT(resolution_wallclock_hours) >= 100
ORDER BY median_hours DESC, issue_type_id
LIMIT 10;
