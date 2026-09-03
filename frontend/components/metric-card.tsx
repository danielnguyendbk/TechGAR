import type { ReactNode } from 'react';

import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card';

export function MetricCard({ icon, label, value, suffix, detail }: {
  readonly icon: ReactNode;
  readonly label: string;
  readonly value: string;
  readonly suffix?: string;
  readonly detail: string;
}) {
  return (
    <Card size="sm" className="gap-2">
      <CardHeader className="grid-cols-[1fr_auto]">
        <CardDescription>{label}</CardDescription>
        <span className="text-[#357a53] [&>svg]:size-4">{icon}</span>
      </CardHeader>
      <CardContent>
        <p className="font-heading text-2xl font-semibold tracking-tight">{value}<span className="ml-1 text-xs font-medium text-muted-foreground">{suffix}</span></p>
        <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}
