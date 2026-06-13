import type { ReactNode } from 'react';

export default function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`relative rounded-brutal border-4 border-ink bg-foam-50 p-5 shadow-brutal ${className}`}
    >
      {/* corner foam-claw accent — cyan pops on the beige card */}
      <span
        aria-hidden
        className="pointer-events-none absolute right-3 top-3 h-3 w-3 rotate-45 border-r-4 border-t-4 border-pink"
      />
      {(title || actions) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && (
              <h3 className="font-display text-lg font-extrabold uppercase tracking-tight text-ink">
                {title}
              </h3>
            )}
            {subtitle && <p className="mt-0.5 text-xs font-medium text-prussian-900">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
