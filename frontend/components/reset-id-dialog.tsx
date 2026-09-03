'use client';

import { useState } from 'react';
import { RotateCcw, ShieldAlert } from 'lucide-react';

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { runtimeClient } from '@/lib/api/runtime-client';
import { parkingStore } from '@/lib/stores/parking-store';

export function ResetIdDialog() {
  const [includeSessions, setIncludeSessions] = useState(false);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function reset(): Promise<void> {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      const response = await runtimeClient.resetIdentities(includeSessions);
      parkingStore.resetLocal();
      setResult(`Đã reset ${response.retired_identities} Global ID${response.include_sessions ? ' và các phiên liên quan' : ''}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể reset Global ID');
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-2">
      <AlertDialog>
        <AlertDialogTrigger render={<Button variant="outline" className="w-full justify-start" />}>
          <RotateCcw className="size-4" /> Reset Global ID
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className="bg-[#fff0e9] text-[#943d2f]"><ShieldAlert /></AlertDialogMedia>
            <AlertDialogTitle>Xác nhận reset danh tính?</AlertDialogTitle>
            <AlertDialogDescription>
              Registry và tracker cục bộ sẽ được làm sạch. Thao tác chỉ được gửi một lần sau khi xác nhận.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <label htmlFor="include-sessions" className="flex min-h-11 items-center gap-3 rounded-lg border p-3 text-sm">
            <Checkbox id="include-sessions" checked={includeSessions} onCheckedChange={(checked) => setIncludeSessions(checked === true)} />
            Kết thúc cả các phiên tài xế đang liên kết
          </label>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={pending}>Hủy</AlertDialogCancel>
            <AlertDialogAction disabled={pending} onClick={() => { void reset(); }}>
              {pending ? 'Đang reset…' : 'Xác nhận reset'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {result && <output className="block rounded-lg bg-[#eef7ef] px-3 py-2 text-xs text-[#285c3e]">{result}</output>}
      {error && <p role="alert" className="rounded-lg bg-[#fff0e9] px-3 py-2 text-xs text-[#943d2f]">{error}</p>}
    </div>
  );
}
