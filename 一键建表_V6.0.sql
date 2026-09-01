-- 股票决策辅助 Agent V6.0：邮箱账号永久保存
-- 在 Supabase 的 SQL Editor 中整段粘贴并点击 Run。重复执行也是安全的。

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table if not exists public.risk_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  profile_data jsonb not null default '{}'::jsonb,
  risk_score integer not null default 0,
  risk_level text not null default '',
  version integer not null default 1,
  completed_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.risk_drafts (
  user_id uuid primary key references auth.users(id) on delete cascade,
  answers jsonb not null default '{}'::jsonb,
  current_index integer not null default 0,
  updated_at timestamptz not null default now()
);

create table if not exists public.stock_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  market text not null,
  stock_code text not null,
  stock_name text not null default '',
  principal_rmb numeric(20,3) not null default 0,
  principal_source text not null default '尚未记录实际投入',
  total_shares numeric(24,6),
  shares_complete boolean not null default false,
  messages jsonb not null default '[]'::jsonb,
  recorded_event_ids jsonb not null default '[]'::jsonb,
  latest_summary jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, market, stock_code),
  constraint stock_sessions_principal_nonnegative check (principal_rmb >= 0),
  constraint stock_sessions_shares_nonnegative check (total_shares is null or total_shares >= 0)
);

create table if not exists public.analysis_snapshots (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  stock_session_id uuid not null references public.stock_sessions(id) on delete cascade,
  event_id text not null,
  summary jsonb not null default '{}'::jsonb,
  snapshot_data jsonb not null,
  created_at timestamptz not null default now(),
  unique (user_id, event_id)
);

create table if not exists public.position_transactions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  stock_session_id uuid not null references public.stock_sessions(id) on delete cascade,
  transaction_type text not null check (transaction_type in ('initial', 'add')),
  input_method text not null check (input_method in ('amount', 'shares')),
  trade_date date not null,
  shares numeric(24,6),
  price_native numeric(24,6),
  currency text not null default '人民币元',
  fx_rate numeric(18,8),
  fees_rmb numeric(20,3) not null default 0,
  principal_rmb numeric(20,3) not null,
  note text not null default '',
  created_at timestamptz not null default now(),
  constraint position_principal_positive check (principal_rmb > 0),
  constraint position_fees_nonnegative check (fees_rmb >= 0),
  constraint position_shares_positive check (shares is null or shares > 0),
  constraint position_price_positive check (price_native is null or price_native > 0),
  constraint position_fx_positive check (fx_rate is null or fx_rate > 0)
);

-- 如果你曾运行过早期测试版 SQL，下面这些语句会补齐新字段，不会删除旧数据。
alter table public.risk_profiles add column if not exists profile_data jsonb not null default '{}'::jsonb;
alter table public.risk_profiles add column if not exists risk_score integer not null default 0;
alter table public.risk_profiles add column if not exists risk_level text not null default '';
alter table public.risk_profiles add column if not exists version integer not null default 1;
alter table public.risk_profiles add column if not exists completed_at timestamptz not null default now();
alter table public.risk_profiles add column if not exists updated_at timestamptz not null default now();

alter table public.risk_drafts add column if not exists answers jsonb not null default '{}'::jsonb;
alter table public.risk_drafts add column if not exists current_index integer not null default 0;
alter table public.risk_drafts add column if not exists updated_at timestamptz not null default now();

alter table public.stock_sessions add column if not exists stock_name text not null default '';
alter table public.stock_sessions add column if not exists principal_rmb numeric(20,3) not null default 0;
alter table public.stock_sessions add column if not exists principal_source text not null default '尚未记录实际投入';
alter table public.stock_sessions add column if not exists total_shares numeric(24,6);
alter table public.stock_sessions add column if not exists shares_complete boolean not null default false;
alter table public.stock_sessions add column if not exists messages jsonb not null default '[]'::jsonb;
alter table public.stock_sessions add column if not exists recorded_event_ids jsonb not null default '[]'::jsonb;
alter table public.stock_sessions add column if not exists latest_summary jsonb not null default '{}'::jsonb;
alter table public.stock_sessions add column if not exists created_at timestamptz not null default now();
alter table public.stock_sessions add column if not exists updated_at timestamptz not null default now();

alter table public.analysis_snapshots add column if not exists summary jsonb not null default '{}'::jsonb;
alter table public.analysis_snapshots add column if not exists snapshot_data jsonb;
alter table public.analysis_snapshots add column if not exists created_at timestamptz not null default now();

create unique index if not exists idx_risk_profiles_user_unique on public.risk_profiles(user_id);
create unique index if not exists idx_risk_drafts_user_unique on public.risk_drafts(user_id);
create index if not exists idx_stock_sessions_user on public.stock_sessions(user_id);
create unique index if not exists idx_stock_sessions_user_market_code_unique on public.stock_sessions(user_id, market, stock_code);
create index if not exists idx_analysis_snapshots_user_session on public.analysis_snapshots(user_id, stock_session_id, created_at);
create unique index if not exists idx_analysis_snapshots_user_event_unique on public.analysis_snapshots(user_id, event_id);
create index if not exists idx_position_transactions_user_session on public.position_transactions(user_id, stock_session_id, trade_date);

drop trigger if exists trg_risk_profiles_updated_at on public.risk_profiles;
create trigger trg_risk_profiles_updated_at
before update on public.risk_profiles
for each row execute function public.set_updated_at();

drop trigger if exists trg_risk_drafts_updated_at on public.risk_drafts;
create trigger trg_risk_drafts_updated_at
before update on public.risk_drafts
for each row execute function public.set_updated_at();

drop trigger if exists trg_stock_sessions_updated_at on public.stock_sessions;
create trigger trg_stock_sessions_updated_at
before update on public.stock_sessions
for each row execute function public.set_updated_at();

create or replace function public.apply_position_transaction()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if tg_op = 'INSERT' then
    update public.stock_sessions
    set
      principal_rmb = principal_rmb + new.principal_rmb,
      principal_source = '会话内买入／加仓记录',
      total_shares = case
        when new.shares is null then total_shares
        when principal_rmb = 0 and total_shares is null then new.shares
        when shares_complete then coalesce(total_shares, 0) + new.shares
        else total_shares
      end,
      shares_complete = case
        when new.shares is null then false
        when principal_rmb = 0 and total_shares is null then true
        else shares_complete
      end
    where id = new.stock_session_id and user_id = new.user_id;
    return new;
  end if;

  if tg_op = 'DELETE' then
    update public.stock_sessions
    set
      principal_rmb = greatest(principal_rmb - old.principal_rmb, 0),
      principal_source = case
        when principal_rmb <= old.principal_rmb then '尚未记录实际投入'
        else '会话内买入／加仓记录'
      end,
      total_shares = case
        when principal_rmb <= old.principal_rmb then null
        when old.shares is not null and shares_complete then greatest(coalesce(total_shares, 0) - old.shares, 0)
        else total_shares
      end,
      shares_complete = case
        when principal_rmb <= old.principal_rmb then false
        else shares_complete
      end
    where id = old.stock_session_id and user_id = old.user_id;
    return old;
  end if;

  return null;
end;
$$;

drop trigger if exists trg_apply_position_transaction on public.position_transactions;
create trigger trg_apply_position_transaction
after insert or delete on public.position_transactions
for each row execute function public.apply_position_transaction();

alter table public.risk_profiles enable row level security;
alter table public.risk_drafts enable row level security;
alter table public.stock_sessions enable row level security;
alter table public.analysis_snapshots enable row level security;
alter table public.position_transactions enable row level security;

drop policy if exists risk_profiles_own_rows on public.risk_profiles;
create policy risk_profiles_own_rows on public.risk_profiles
for all to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

drop policy if exists risk_drafts_own_rows on public.risk_drafts;
create policy risk_drafts_own_rows on public.risk_drafts
for all to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

drop policy if exists stock_sessions_own_rows on public.stock_sessions;
create policy stock_sessions_own_rows on public.stock_sessions
for all to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

drop policy if exists analysis_snapshots_own_rows on public.analysis_snapshots;
create policy analysis_snapshots_own_rows on public.analysis_snapshots
for all to authenticated
using (
  (select auth.uid()) = user_id
  and exists (
    select 1 from public.stock_sessions s
    where s.id = stock_session_id and s.user_id = (select auth.uid())
  )
)
with check (
  (select auth.uid()) = user_id
  and exists (
    select 1 from public.stock_sessions s
    where s.id = stock_session_id and s.user_id = (select auth.uid())
  )
);

drop policy if exists position_transactions_own_rows on public.position_transactions;
create policy position_transactions_own_rows on public.position_transactions
for all to authenticated
using (
  (select auth.uid()) = user_id
  and exists (
    select 1 from public.stock_sessions s
    where s.id = stock_session_id and s.user_id = (select auth.uid())
  )
)
with check (
  (select auth.uid()) = user_id
  and exists (
    select 1 from public.stock_sessions s
    where s.id = stock_session_id and s.user_id = (select auth.uid())
  )
);

grant select, insert, update, delete on public.risk_profiles to authenticated;
grant select, insert, update, delete on public.risk_drafts to authenticated;
grant select, insert, update, delete on public.stock_sessions to authenticated;
grant select, insert, update, delete on public.analysis_snapshots to authenticated;
grant select, insert, update, delete on public.position_transactions to authenticated;

select 'V6.0 数据表、自动持仓汇总和用户隔离权限已配置完成' as result;
