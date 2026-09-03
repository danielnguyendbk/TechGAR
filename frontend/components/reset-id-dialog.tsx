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

  async function softReset(): Promise<void> {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      await runtimeClient.softReset();
      setResult('Đã thực hiện Soft Reset (bảo toàn GID, ô đỗ và phiên).');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể soft reset');
    } finally {
      setPending(false);
    }
  }

  async function closeAll(): Promise<void> {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      await runtimeClient.closeAll(true);
      parkingStore.resetLocal();
      setResult('Đã Close-All: đóng tất cả phiên và reset runtime (bảo toàn chuỗi GID).');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể close-all');
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-3">
      <Button
        variant="outline"
        className="w-full justify-start border-[#357a53]/30 hover:bg-[#357a53]/10"
        disabled={pending}
        onClick={() => { void softReset(); }}
      >
        <RotateCcw className="size-4 text-[#357a53]" />
        Soft Reset (giữ GID & xe đỗ)
      </Button>

      <AlertDialog>
        <AlertDialogTrigger render={<Button variant="outline" className="w-full justify-start text-destructive hover:bg-destructive/10" />}>
          <ShieldAlert className="size-4" /> Close-All (Hard Reset)
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className="bg-[#fff0e9] text-[#943d2f]"><ShieldAlert /></AlertDialogMedia>
            <AlertDialogTitle>Xác nhận Close-All toàn hệ thống?</AlertDialogTitle>
            <AlertDialogDescription>
              Tất cả các phiên tài xế sẽ đóng ngay lập tức, bộ nhớ tracking sẽ được làm sạch.
              Chuỗi GID sẽ tiếp tục tăng đơn điệu và KHÔNG bị quay về 0.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={pending}>Hủy</AlertDialogCancel>
            <AlertDialogAction disabled={pending} onClick={() => { void closeAll(); }}>
              {pending ? 'Đang xử lý…' : 'Xác nhận Close-All'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {result && <output className="block rounded-lg bg-[#eef7ef] px-3 py-2 text-xs text-[#285c3e]">{result}</output>}
      {error && <p role="alert" className="rounded-lg bg-[#fff0e9] px-3 py-2 text-xs text-[#943d2f]">{error}</p>}
    </div>
  );
}
